from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def _get_pattern_in_reversed_nested_tuple_format(config: BackendPatternConfig) -> Pattern:
    """
    Return the pattern specified in the given config in the reversed nested tuple format
    used internally in the quantization pattern matching code.

    If the pattern is not a tuple, or the pattern is already specified in the reversed
    nested tuple format, return the pattern as is. Otherwise:

    For 2-tuples (a, b), return (b, a).
    For 3-tuples (a, b, c), return (c, (b, a)).

    For example:
        * Given nn.Linear, return nn.Linear
        * Given (nn.Linear, nn.ReLU), return (nn.ReLU, nn.Linear)
        * Given (nn.Conv2d, nn.BatchNorm2d, nn.ReLU), return
          (nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))

    For context, the reason why this is needed is the user-facing BackendConfig
    API accepts the flat 2-or-3-tuple format in forward order. While this simple
    format handles the vast majority of use cases, it does not handle the more
    complex ones, and so the internal pattern matching code for quantization uses
    the following, more general reversed nested tuple format instead:

        operator = module_type | functional | torch op | native op | MatchAllNode
        Pattern = (operator, Pattern, Pattern, ...) | operator

    In the future, we expect to replace the above complex format with the one used
    by the subgraph rewriter in torch.fx, so we don't have to maintain our own
    complex pattern matching code. Then we won't need this helper function anymore.
    """
    if config._pattern_complex_format is not None:
        return config._pattern_complex_format
    if config.pattern is None:
        raise ValueError("Either 'pattern' or 'pattern_complex_format' must be specified")
    if not isinstance(config.pattern, tuple):
        return config.pattern
    if len(config.pattern) == 2:
        a, b = config.pattern
        return (b, a)
    elif len(config.pattern) == 3:
        a, b, c = config.pattern
        return (c, (b, a))
    else:
        raise ValueError('Expected a tuple with 2 or 3 elements, got: ', config.pattern)