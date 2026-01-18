import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_git import GitConfig, GitVisionConfig
@add_start_docstrings('The bare GIT Model transformer consisting of a CLIP image encoder and text decoder outputting raw hidden-states without any specific head on top.', GIT_START_DOCSTRING)
class GitModel(GitPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = GitEmbeddings(config)
        self.image_encoder = GitVisionModel(config.vision_config)
        self.encoder = GitEncoder(config)
        self.visual_projection = GitProjection(config)
        if config.num_image_with_embedding is not None:
            self.img_temperal_embedding = nn.ParameterList((nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size)) for _ in range(config.num_image_with_embedding)))
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        top_right = torch.full((num_memory, num_tgt + past_key_values_length), float('-inf'), device=tgt.device, dtype=dtype)
        bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device)
        if past_key_values_length > 0:
            tgt_mask = torch.zeros((tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length), dtype=dtype, device=tgt_mask.device)
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)
        full_attention_mask = torch.cat((left, right), dim=1)[None, :]
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError('Memory key padding mask must be a boolean tensor.')
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update
        full_attention_mask = full_attention_mask[:, None, :, :]
        return full_attention_mask

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        """
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> import requests
        >>> from PIL import Image

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")
        >>> model = AutoModel.from_pretrained("microsoft/git-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = "this is an image of two cats"

        >>> inputs = processor(text, images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
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
        seq_length = input_shape[1]
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        projected_visual_features = None
        if pixel_values is not None:
            if pixel_values.ndim == 4:
                visual_features = self.image_encoder(pixel_values).last_hidden_state
            elif pixel_values.ndim == 5:
                visual_features = []
                for frame_idx in range(pixel_values.shape[1]):
                    visual_features_frame = self.image_encoder(pixel_values[:, frame_idx, :, :]).last_hidden_state
                    visual_features_frame += self.img_temperal_embedding[frame_idx]
                    visual_features.append(visual_features_frame)
                visual_features = torch.cat(visual_features, dim=1)
            else:
                raise ValueError('pixel_values must be of rank 4 or 5')
            projected_visual_features = self.visual_projection(visual_features)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        if projected_visual_features is None:
            projected_visual_features = torch.zeros((embedding_output.shape[0], 0, embedding_output.shape[2]), dtype=embedding_output.dtype, device=embedding_output.device)
        projected_visual_features = projected_visual_features.repeat(embedding_output.size(0) // projected_visual_features.size(0), 1, 1)
        hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)
        combined_attention_mask = self.create_attention_mask(tgt=embedding_output, memory=projected_visual_features, tgt_mask=tgt_mask, past_key_values_length=past_key_values_length)
        if attention_mask is not None:
            expanded_attn_mask = _prepare_4d_attention_mask(attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(embedding_output.device)
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:
                combined_attention_mask[:, :, -input_shape[1]:, -input_shape[1]:] += expanded_attn_mask
        encoder_outputs = self.encoder(hidden_states, attention_mask=combined_attention_mask, head_mask=head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, pixel_values_present=pixel_values is not None)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithPast(last_hidden_state=sequence_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)