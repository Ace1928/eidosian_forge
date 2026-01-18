from collections import OrderedDict
from typing import Mapping
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class MobileViTV2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse('1.11')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == 'image-classification':
            return OrderedDict([('logits', {0: 'batch'})])
        else:
            return OrderedDict([('last_hidden_state', {0: 'batch'}), ('pooler_output', {0: 'batch'})])

    @property
    def atol_for_validation(self) -> float:
        return 0.0001