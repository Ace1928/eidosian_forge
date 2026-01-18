from collections import OrderedDict
from typing import Dict, Any
from torch.ao.quantization.utils import Pattern
from ..fake_quantize import FixedQParamsFakeQuantize
from ..observer import ObserverBase
import copy
def _register_fusion_pattern(pattern):

    def insert(fn):
        _DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert