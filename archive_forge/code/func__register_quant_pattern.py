from collections import OrderedDict
from typing import Dict, Any
from torch.ao.quantization.utils import Pattern
from ..fake_quantize import FixedQParamsFakeQuantize
from ..observer import ObserverBase
import copy
def _register_quant_pattern(pattern, fixed_qparams_observer=None):

    def insert(fn):
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            _DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn
    return insert