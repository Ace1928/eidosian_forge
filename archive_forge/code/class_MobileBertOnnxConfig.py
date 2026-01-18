from collections import OrderedDict
from typing import Mapping
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class MobileBertOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == 'multiple-choice':
            dynamic_axis = {0: 'batch', 1: 'choice', 2: 'sequence'}
        else:
            dynamic_axis = {0: 'batch', 1: 'sequence'}
        return OrderedDict([('input_ids', dynamic_axis), ('attention_mask', dynamic_axis), ('token_type_ids', dynamic_axis)])