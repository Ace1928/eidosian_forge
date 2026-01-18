import math
from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_outputs import (
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig
@add_start_docstrings('\n    VAN Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', VAN_START_DOCSTRING)
class VanForImageClassification(VanPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.van = VanModel(config)
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.van(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)