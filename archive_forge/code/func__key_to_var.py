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
@classmethod
def _key_to_var(cls, tx, key, **options):
    from .builder import VariableBuilder
    if istensor(key):
        return VariableBuilder(tx, GlobalWeakRefSource(global_key_name(key)))(key)
    else:
        assert ConstantVariable.is_literal(key)
        return ConstantVariable.create(key, **options)