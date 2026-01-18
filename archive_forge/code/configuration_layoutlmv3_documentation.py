from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import logging

        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            processor ([`ProcessorMixin`]):
                The processor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2).
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the processor will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Any]: holding the kwargs to provide to the model's forward function
        