from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig
class DPREncoder(DPRPreTrainedModel):
    base_model_prefix = 'bert_model'

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.bert_model = BertModel(config, add_pooling_layer=False)
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        self.post_init()

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor]=None, token_type_ids: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.hidden_size