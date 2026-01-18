import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _copy_attr
class LeafModuleAwareTracer(fx.Tracer):
    """
    An fx.Tracer that allows the user to specify a set of leaf modules, i.e.
    modules that are not to be traced through. The resulting graph ends up
    having single nodes referencing calls to the leaf modules' forward methods.
    """

    def __init__(self, *args, **kwargs):
        self.leaf_modules = {}
        if 'leaf_modules' in kwargs:
            leaf_modules = kwargs.pop('leaf_modules')
            self.leaf_modules = leaf_modules
        super().__init__(*args, **kwargs)

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if isinstance(m, tuple(self.leaf_modules)):
            return True
        return super().is_leaf_module(m, module_qualname)