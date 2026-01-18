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
@add_start_docstrings('\n    Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]\n    token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.\n    ', VILT_START_DOCSTRING)
class ViltForImageAndTextRetrieval(ViltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.vilt = ViltModel(config)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        >>> model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, :].item()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vilt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, image_embeds=image_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooler_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.rank_output(pooler_output)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            raise NotImplementedError('Training is not yet supported.')
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)