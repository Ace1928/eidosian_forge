import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
@property
def _underlying_items(self):
    underlying_items = set()
    for current_item in self.items:
        assert current_item not in underlying_items, 'Items modeling set invariant violated'
        underlying_items.add(self._as_set_element(current_item))
    return underlying_items