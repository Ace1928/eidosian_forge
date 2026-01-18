import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def _get_default_policy(allow_list=None):
    _default_allow_list = ['xformers.efficient_attention_forward_cutlass.default', 'xformers_flash.flash_fwd.default', 'aten.addmm.default', 'aten.mm.default']
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(mode, func, *args, **kwargs):
        return str(func) in allow_list
    return _default_policy