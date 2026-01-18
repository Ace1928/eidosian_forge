import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_xlnet import XLNetConfig
@add_start_docstrings('\n    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.\n    for GLUE tasks.\n    ', XLNET_START_DOCSTRING)
class XLNetForSequenceClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForSequenceClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForSequenceClassificationOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetForSequenceClassificationOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)