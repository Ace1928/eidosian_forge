import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_altclip import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig
class AltCLIPTextModel(AltCLIPPreTrainedModel):
    config_class = AltCLIPTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = AltRobertaModel(config, add_pooling_layer=False)
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.roberta.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.roberta.embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None) -> nn.Embedding:
        return super().resize_token_embeddings(new_num_tokens)

    @add_start_docstrings_to_model_forward(ALTCLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndProjection, config_class=AltCLIPTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPoolingAndProjection]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AltCLIPTextModel

        >>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> texts = ["it's a cat", "it's a dog"]

        >>> inputs = processor(text=texts, padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.pre_LN(sequence_output)
        projection_state = self.transformation(sequence_output)
        pooler_output = projection_state[:, 0]
        if not return_dict:
            return (projection_state, pooler_output) + outputs[2:4]
        return BaseModelOutputWithPoolingAndProjection(last_hidden_state=projection_state, pooler_output=pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)