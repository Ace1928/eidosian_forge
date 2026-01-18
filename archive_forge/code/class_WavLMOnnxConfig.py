import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class WavLMOnnxConfig(HubertOnnxConfig):
    DEFAULT_ONNX_OPSET = 12

    def patch_model_for_export(self, model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None) -> 'ModelPatcher':
        return WavLMModelPatcher(self, model, model_kwargs=model_kwargs)