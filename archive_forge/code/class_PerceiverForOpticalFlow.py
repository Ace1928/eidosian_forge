import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
@add_start_docstrings('\nExample use of Perceiver for optical flow, for tasks such as Sintel and KITTI. [`PerceiverForOpticalFlow`] uses\n[`~models.perceiver.modeling_perceiver.PerceiverImagePreprocessor`] (with *prep_type="patches"*) to preprocess the\ninput images, and [`~models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder`] to decode the latent\nrepresentation of [`PerceiverModel`].\n\nAs input, one concatenates 2 subsequent frames along the channel dimension and extract a 3 x 3 patch around each pixel\n(leading to 3 x 3 x 3 x 2 = 54 values for each pixel). Fixed Fourier position encodings are used to encode the position\nof each pixel in the patch. Next, one applies the Perceiver encoder. To decode, one queries the latent representation\nusing the same encoding used for the input.\n', PERCEIVER_START_DOCSTRING)
class PerceiverForOpticalFlow(PerceiverPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        fourier_position_encoding_kwargs_preprocessor = {'num_bands': 64, 'max_resolution': config.train_size, 'sine_only': False, 'concat_pos': True}
        fourier_position_encoding_kwargs_decoder = {'concat_pos': True, 'max_resolution': config.train_size, 'num_bands': 64, 'sine_only': False}
        image_preprocessor = PerceiverImagePreprocessor(config, prep_type='patches', spatial_downsample=1, conv_after_patching=True, conv_after_patching_in_channels=54, temporal_downsample=2, position_encoding_type='fourier', fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor)
        self.perceiver = PerceiverModel(config, input_preprocessor=image_preprocessor, decoder=PerceiverOpticalFlowDecoder(config, num_channels=image_preprocessor.num_channels, output_image_shape=config.train_size, rescale_factor=100.0, use_query_residual=False, output_num_channels=2, position_encoding_type='fourier', fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder))
        self.post_init()

    @add_start_docstrings_to_model_forward(PERCEIVER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=PerceiverClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, inputs: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, PerceiverClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the optical flow loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import PerceiverForOpticalFlow
        >>> import torch

        >>> model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")

        >>> # in the Perceiver IO paper, the authors extract a 3 x 3 patch around each pixel,
        >>> # leading to 3 x 3 x 3 = 27 values for each pixel (as each pixel also has 3 color channels)
        >>> # patches have shape (batch_size, num_frames, num_channels, height, width)
        >>> # the authors train on resolutions of 368 x 496
        >>> patches = torch.randn(1, 2, 27, 368, 496)
        >>> outputs = model(inputs=patches)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 368, 496, 2]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.perceiver(inputs=inputs, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        if labels is not None:
            raise NotImplementedError('Optical flow training is not yet supported')
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return PerceiverClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)