from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...file_utils import (
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoBackbone
from .configuration_depth_anything import DepthAnythingConfig
@add_start_docstrings('\n    Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.\n    ', DEPTH_ANYTHING_START_DOCSTRING)
class DepthAnythingForDepthEstimation(DepthAnythingPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.neck = DepthAnythingNeck(config)
        self.head = DepthAnythingDepthEstimationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(DEPTH_ANYTHING_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        >>> model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        hidden_states = outputs.feature_maps
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        predicted_depth = self.head(hidden_states, patch_height, patch_width)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented yet')
        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return DepthEstimatorOutput(loss=loss, predicted_depth=predicted_depth, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)