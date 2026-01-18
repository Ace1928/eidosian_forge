import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
@FunctionIdSet
def _numpy_function_ids() -> Dict[int, str]:
    rv = dict()
    for mod in NP_SUPPORTED_MODULES:
        rv.update({id(v): f'{mod.__name__}.{k}' for k, v in mod.__dict__.items() if callable(v) and (getattr(v, '__module__', None) or mod.__name__) == mod.__name__})
    return rv