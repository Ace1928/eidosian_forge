from collections import OrderedDict
from typing import Mapping
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
class SwinOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse('1.11')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'})])

    @property
    def atol_for_validation(self) -> float:
        return 0.0001