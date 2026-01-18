from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig
class DPRSpanPredictor(DPRPreTrainedModel):
    base_model_prefix = 'encoder'

    def __init__(self, config: DPRConfig):
        super().__init__(config)
        self.encoder = DPREncoder(config)
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        self.post_init()

    def forward(self, input_ids: Tensor, attention_mask: Tensor, inputs_embeds: Optional[Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        n_passages, sequence_length = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        outputs = self.encoder(input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])
        start_logits = start_logits.view(n_passages, sequence_length)
        end_logits = end_logits.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)
        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]
        return DPRReaderOutput(start_logits=start_logits, end_logits=end_logits, relevance_logits=relevance_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)