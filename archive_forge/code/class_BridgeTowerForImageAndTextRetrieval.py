import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
@add_start_docstrings('\n    BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the\n    [CLS] token) for image-to-text matching.\n    ', BRIDGETOWER_START_DOCSTRING)
class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bridgetower = BridgeTowerModel(config)
        self.itm_score = BridgeTowerITMHead(config.hidden_size * 2)
        self.post_init()

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Labels for computing the image-text matching loss. 0 means the pairs don't match and 1 means they match.
            The pairs with 0 will be skipped for calculation.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, 1].item()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bridgetower(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, image_embeds=image_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooler_output = outputs.pooler_output if return_dict else outputs[2]
        logits = self.itm_score(pooler_output)
        itm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            itm_loss = loss_fct(logits, labels)
        if not return_dict:
            output = tuple(logits)
            return (itm_loss,) + output if itm_loss is not None else output
        return SequenceClassifierOutput(loss=itm_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)