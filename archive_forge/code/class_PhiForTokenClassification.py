import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_phi import PhiConfig
@add_start_docstrings('\n    PhiModel with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', PHI_START_DOCSTRING)
class PhiForTokenClassification(PhiPreTrainedModel):

    def __init__(self, config: PhiConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PhiModel(config)
        if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, 'hidden_dropout') and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_outputs = self.model(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = model_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length))
        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=model_outputs.hidden_states, attentions=model_outputs.attentions)