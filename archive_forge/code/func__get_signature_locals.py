import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def _get_signature_locals(f: Callable, loc: Dict[str, Any]) -> Dict[str, Any]:
    """ Get local keyword arguments

    Example::

    >> def f(self, a, b=9):
           pass
    >> loc = {"a": 6, "c": 7}
    >> _get_signature_locals(f, loc)
    {"a": 6}
    """
    return {k: v for k, v in loc.items() if k in signature(f).parameters}