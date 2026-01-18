import math
import os
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import (
from ...modeling_utils import (
from ...utils import logging
from .configuration_markuplm import MarkupLMConfig
@add_start_docstrings('MarkupLM Model with a `token_classification` head on top.', MARKUPLM_START_DOCSTRING)
class MarkupLMForTokenClassification(MarkupLMPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, xpath_tags_seq: Optional[torch.Tensor]=None, xpath_subs_seq: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
        >>> processor.parse_html = False
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/markuplm-base", num_labels=7)

        >>> nodes = ["hello", "world"]
        >>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"]
        >>> node_labels = [1, 2]
        >>> encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**encoding)

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.markuplm(input_ids, xpath_tags_seq=xpath_tags_seq, xpath_subs_seq=xpath_subs_seq, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.num_labels), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)