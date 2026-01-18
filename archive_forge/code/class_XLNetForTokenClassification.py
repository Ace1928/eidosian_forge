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
@add_start_docstrings('\n    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', XLNET_START_DOCSTRING)
class XLNetForTokenClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForTokenClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetForTokenClassificationOutput(loss=loss, logits=logits, mems=outputs.mems, hidden_states=outputs.hidden_states, attentions=outputs.attentions)