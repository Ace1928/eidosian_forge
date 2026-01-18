import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_glpn import GLPNConfig
@add_start_docstrings('GLPN Model transformer with a lightweight depth estimation head on top e.g. for KITTI, NYUv2.', GLPN_START_DOCSTRING)
class GLPNForDepthEstimation(GLPNPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.glpn = GLPNModel(config)
        self.decoder = GLPNDecoder(config)
        self.head = GLPNDepthEstimationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(GLPN_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        """
        labels (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, GLPNForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
        >>> model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

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
        outputs = self.glpn(pixel_values, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        out = self.decoder(hidden_states)
        predicted_depth = self.head(out)
        loss = None
        if labels is not None:
            loss_fct = SiLogLoss()
            loss = loss_fct(predicted_depth, labels)
        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return DepthEstimatorOutput(loss=loss, predicted_depth=predicted_depth, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)