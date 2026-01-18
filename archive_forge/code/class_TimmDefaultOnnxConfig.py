import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class TimmDefaultOnnxConfig(ViTOnnxConfig):
    ATOL_FOR_VALIDATION = 0.001
    DEFAULT_ONNX_OPSET = 12

    def rename_ambiguous_inputs(self, inputs):
        model_inputs = {}
        model_inputs['x'] = inputs['pixel_values']
        return model_inputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        return {'x': 'pixel_values'}