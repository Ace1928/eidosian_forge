import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def _process_model_after_weight_loading(self, model: 'PreTrainedModel', **kwargs):
    model.is_loaded_in_4bit = True
    model.is_4bit_serializable = self.is_serializable
    return model