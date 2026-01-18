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
def _call_hasattr_customobj(self, tx, name: str) -> 'VariableTracker':
    """Shared method between DataClassVariable and CustomizedDictVariable where items are attrs"""
    if name in self.items or hasattr(self.user_cls, name):
        return ConstantVariable(True)
    elif istype(self.mutable_local, MutableLocal) and self.source is None:
        return ConstantVariable(False)
    elif self.mutable_local is None and self.source:
        try:
            example = tx.output.root_tx.get_example_value(self.source)
            install_guard(AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR))
            return ConstantVariable(hasattr(example, name))
        except KeyError:
            pass
    unimplemented(f'hasattr({self.__class__.__name__}, {name}) {self.mutable_local} {self.source}')