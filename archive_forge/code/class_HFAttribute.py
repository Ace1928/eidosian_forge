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
class HFAttribute(HFProxy):

    def __init__(self, root, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None
        if hasattr(self.root, '_metadata'):
            self.install_metadata(getattr(self.root._metadata, attr))

    @property
    def node(self):
        if self._node is None:
            self._node = self.tracer.create_proxy('call_function', builtins.getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)