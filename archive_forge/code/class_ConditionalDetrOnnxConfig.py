from collections import OrderedDict
from typing import Mapping
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
class ConditionalDetrOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse('1.11')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'}), ('pixel_mask', {0: 'batch'})])

    @property
    def atol_for_validation(self) -> float:
        return 1e-05

    @property
    def default_onnx_opset(self) -> int:
        return 12