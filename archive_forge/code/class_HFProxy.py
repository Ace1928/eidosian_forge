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
class HFProxy(Proxy):
    """
    Proxy that uses metadata to handle data-dependent control-flow.
    """

    def install_metadata(self, metadata):
        self._metadata = metadata

    @property
    def shape(self):
        return self.tracer.create_proxy('call_method', 'size', (self,), {})

    @property
    def device(self):
        return MetaDeviceAttribute(self, 'device')

    def __len__(self):
        if hasattr(self, '_metadata') and self._metadata is not None:
            return len(self._metadata)
        return super().__len__()

    def __bool__(self):
        if hasattr(self, '_metadata') and self._metadata is not None:
            return self._metadata
        return super().__bool__()

    def __getattr__(self, k):
        if k == '_metadata':
            return self.__getattribute__(k)
        return HFAttribute(self, k)

    def __setitem__(self, indices, values):
        return self.tracer.create_proxy('call_function', operator.setitem, (self, indices, values), {})

    def __contains__(self, key):
        if hasattr(self, '_metadata') and self._metadata is not None:
            return key in self._metadata
        return super().__contains__(key)