import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vilt import ViltConfig
@add_start_docstrings('\n    Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]\n    token) for visual question answering, e.g. for VQAv2.\n    ', VILT_START_DOCSTRING)
class ViltForQuestionAnswering(ViltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size * 2), nn.LayerNorm(config.hidden_size * 2), nn.GELU(), nn.Linear(config.hidden_size * 2, config.num_labels))
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
            all answers that are applicable for a given example in the batch, or a soft encoding indicating which
            answers are applicable, where 1.0 is the highest score.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForQuestionAnswering
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are there?"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        >>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)
        >>> logits = outputs.logits
        >>> idx = logits.argmax(-1).item()
        >>> print("Predicted answer:", model.config.id2label[idx])
        Predicted answer: 2
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vilt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, image_embeds=image_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooler_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooler_output)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) * labels.shape[1]
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)