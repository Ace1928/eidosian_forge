import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def _insert_module_as_submodule(self, mod: nn.Module) -> str:
    """
        Helper method which tries to insert a module that was not declared as submodule.
        """
    if self._stateless_mod_instanciation_depends_on_proxies(mod):
        return ''
    idx = 0
    mod_name = mod.__class__.__name__.lower()
    path = f'{mod_name}_{idx}'
    already_inserted = False
    while hasattr(self.root, path):
        if getattr(self.root, path) is mod:
            already_inserted = True
            break
        path = f'{mod_name}_{idx}'
        idx += 1
    if not already_inserted:
        self.root.add_module(path, mod)
    return path