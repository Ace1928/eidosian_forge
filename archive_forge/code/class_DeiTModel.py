import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_deit import DeiTConfig
@add_start_docstrings('The bare DeiT Model transformer outputting raw hidden-states without any specific head on top.', DEIT_START_DOCSTRING)
class DeiTModel(DeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig, add_pooling_layer: bool=True, use_mask_token: bool=False) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = DeiTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DeiTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = DeiTPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> DeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, bool_masked_pos: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)