from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def _get_fuser_method_in_reversed_nested_tuple_format(config: BackendPatternConfig) -> Callable:
    """
    Return the fuser method specified in the given config in the reversed nested
    tuple format used internally in the quantization pattern matching code.

    If pattern is specified in the reversed nested tuple format, we assume the
    fuser method is also specified in this format and simply return it as is.
    Otherwise, we convert the fuser method as follows:

        * Given f(is_qat, conv, relu), return f'(is_qat, relu, conv)
        * Given f(is_qat, conv, bn, relu), return f'(is_qat, relu, bn_conv),
          where bn_conv is a 2-tuple (bn, conv)

    The first argument of a fuser method is always `is_qat` and is not affected
    in the conversion. We currently only support functions with 3 or 4 arguments.
    """
    assert config.fuser_method is not None
    if config._pattern_complex_format is not None:
        return config.fuser_method
    if not isinstance(config.pattern, tuple):
        raise ValueError('Expected pattern to be a tuple, got: ', config.pattern)
    if len(config.pattern) == 2:
        return _reverse2(config.fuser_method)
    elif len(config.pattern) == 3:
        return _reverse3(config.fuser_method)
    else:
        raise ValueError('Expected a tuple with 2 or 3 elements, got: ', config.pattern)