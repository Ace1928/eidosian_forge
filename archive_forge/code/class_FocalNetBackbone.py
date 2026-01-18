import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig
@add_start_docstrings('\n    FocalNet backbone, to be used with frameworks like X-Decoder.\n    ', FOCALNET_START_DOCSTRING)
class FocalNetBackbone(FocalNetPreTrainedModel, BackboneMixin):

    def __init__(self, config: FocalNetConfig):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = [config.embed_dim] + config.hidden_sizes
        self.focalnet = FocalNetModel(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny-lrf")
        >>> model = AutoBackbone.from_pretrained("microsoft/focalnet-tiny-lrf")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.focalnet(pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.reshaped_hidden_states
        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output
        return BackboneOutput(feature_maps=feature_maps, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=None)