import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
@add_start_docstrings('\n    DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.\n    ', DPT_START_DOCSTRING)
class DPTForSemanticSegmentation(DPTPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.dpt = DPTModel(config, add_pooling_layer=False)
        self.neck = DPTNeck(config)
        self.head = DPTSemanticSegmentationHead(config)
        self.auxiliary_head = DPTAuxiliaryHead(config) if config.use_auxiliary_head else None
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SemanticSegmenterOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large-ade")
        >>> model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.dpt(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        if not self.config.is_hybrid:
            hidden_states = [feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices]
        else:
            backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
            backbone_hidden_states.extend((feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices[2:]))
            hidden_states = backbone_hidden_states
        hidden_states = self.neck(hidden_states=hidden_states)
        logits = self.head(hidden_states)
        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(hidden_states[-1])
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError('The number of labels should be greater than one')
            else:
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                if auxiliary_logits is not None:
                    upsampled_auxiliary_logits = nn.functional.interpolate(auxiliary_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                main_loss = loss_fct(upsampled_logits, labels)
                auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
                loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)