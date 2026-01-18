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
class KeyedJaggedTensorVariable(UserDefinedObjectVariable):

    @staticmethod
    def is_matching_object(obj):
        mod = sys.modules.get('torchrec.sparse.jagged_tensor')
        return mod is not None and type(obj) is mod.KeyedJaggedTensor

    def __init__(self, value, **kwargs):
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        assert type(value) is KeyedJaggedTensor
        super().__init__(value, **kwargs)

    def var_getattr(self, tx, name):
        if torch._dynamo.config.force_unspec_int_unbacked_size_like_on_torchrec_kjt and self.source is not None and (name in ('_length_per_key', '_offset_per_key')):
            with TracingContext.patch(force_unspec_int_unbacked_size_like=True):
                return super().var_getattr(tx, name)
        return super().var_getattr(tx, name)