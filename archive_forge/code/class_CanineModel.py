import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
@add_start_docstrings('The bare CANINE Model transformer outputting raw hidden-states without any specific head on top.', CANINE_START_DOCSTRING)
class CanineModel(CaninePreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1
        self.char_embeddings = CanineEmbeddings(config)
        self.initial_char_encoder = CanineEncoder(shallow_config, local=True, always_attend_to_first_position=False, first_position_attends_to_all=False, attend_from_chunk_width=config.local_transformer_stride, attend_from_chunk_stride=config.local_transformer_stride, attend_to_chunk_width=config.local_transformer_stride, attend_to_chunk_stride=config.local_transformer_stride)
        self.chars_to_molecules = CharactersToMolecules(config)
        self.encoder = CanineEncoder(config)
        self.projection = ConvProjection(config)
        self.final_char_encoder = CanineEncoder(shallow_config)
        self.pooler = CaninePooler(config) if add_pooling_layer else None
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        batch_size, from_seq_length = (from_tensor.shape[0], from_tensor.shape[1])
        to_seq_length = to_mask.shape[1]
        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)
        mask = broadcast_ones * to_mask
        return mask

    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(poolable_char_mask.float())
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)
        return molecule_attention_mask

    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""
        rate = self.config.downsampling_rate
        molecules_without_extra_cls = molecules[:, 1:, :]
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(last_molecule, repeats=remainder_length + rate, dim=-2)
        return torch.cat([repeated, remainder_repeated], dim=-2)

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CanineModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CanineModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        molecule_attention_mask = self._downsample_attention_mask(attention_mask, downsampling_rate=self.config.downsampling_rate)
        extended_molecule_attention_mask: torch.Tensor = self.get_extended_attention_mask(molecule_attention_mask, (batch_size, molecule_attention_mask.shape[-1]))
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        input_char_embeddings = self.char_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        char_attention_mask = self._create_3d_attention_mask_from_input_mask(input_ids if input_ids is not None else inputs_embeds, attention_mask)
        init_chars_encoder_outputs = self.initial_char_encoder(input_char_embeddings, attention_mask=char_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        input_char_encoding = init_chars_encoder_outputs.last_hidden_state
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)
        encoder_outputs = self.encoder(init_molecule_encoding, attention_mask=extended_molecule_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        molecule_sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(molecule_sequence_output) if self.pooler is not None else None
        repeated_molecules = self._repeat_molecules(molecule_sequence_output, char_seq_length=input_shape[-1])
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)
        sequence_output = self.projection(concat)
        final_chars_encoder_outputs = self.final_char_encoder(sequence_output, attention_mask=extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        sequence_output = final_chars_encoder_outputs.last_hidden_state
        if output_hidden_states:
            deep_encoder_hidden_states = encoder_outputs.hidden_states if return_dict else encoder_outputs[1]
            all_hidden_states = all_hidden_states + init_chars_encoder_outputs.hidden_states + deep_encoder_hidden_states + final_chars_encoder_outputs.hidden_states
        if output_attentions:
            deep_encoder_self_attentions = encoder_outputs.attentions if return_dict else encoder_outputs[-1]
            all_self_attentions = all_self_attentions + init_chars_encoder_outputs.attentions + deep_encoder_self_attentions + final_chars_encoder_outputs.attentions
        if not return_dict:
            output = (sequence_output, pooled_output)
            output += tuple((v for v in [all_hidden_states, all_self_attentions] if v is not None))
            return output
        return CanineModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=all_hidden_states, attentions=all_self_attentions)