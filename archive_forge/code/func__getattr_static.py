import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import sys
import threading
import types
from typing import Dict, List
import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, RandomValueSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable
def _getattr_static(self, name):
    if isinstance(self.value, torch.nn.Module) or '__slots__' in self.value.__class__.__dict__ or type(self.value) == threading.local:
        subobj = getattr(self.value, name)
    else:
        subobj = inspect.getattr_static(self.value, name)
    return subobj