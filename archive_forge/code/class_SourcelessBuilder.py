import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import operator
import re
import sys
import types
from typing import List, NamedTuple, Optional, Union
import torch
from torch import SymInt
from torch._guards import GuardSource, TracingContext
from torch._ops import HigherOrderOperator
from torch._streambase import _EventBase, _StreamBase
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.immutable_collections import immutable_list
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import TensorWeakRef
from .. import config, mutation_guard, replay_record, skipfiles, trace_rules
from ..allowed_functions import (
from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..side_effects import SideEffects
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
from .dicts import (
from .distributed import (
from .functions import (
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lazy import LazyVariableTracker
from .lists import (
from .misc import (
from .nn_module import FSDPManagedNNModuleVariable, UnspecializedNNModuleVariable
from .optimizer import OptimizerVariable
from .tensor import (
from .torch import torch_special_class_types, TorchVariable
from .torch_function import build_torch_function_fn, TensorWithTFOverrideVariable
from .user_defined import (
class SourcelessBuilder:
    """
    Like builder, but stateless and does not require a source. Useful for simple type->VT objects, or objects
    that are being created/evaporated during inlining (ex: consider a locally made list of tensors we then iterate over
    .), such a list should not show up as an artifact from inputs, nor in reconstruction, nor in the graph. However,
    there may be reasons to represent it as a ListVariable internally.

    NOTE - Objects produced here are born UNGUARDED due to the nature of sources!

    NOTE - This class is very new! It will have some rough edges, but it was created to stem the bleeding of giant
    if/else type->VariableTracker trees that were cropping up all over dynamo.
    """

    def __call__(self, tx, value) -> VariableTracker:
        if isinstance(value, VariableTracker):
            return value
        if isinstance(value, dataclasses._HAS_DEFAULT_FACTORY_CLASS):
            return UserDefinedObjectVariable(value)
        if ConstantVariable.is_literal(value):
            return SourcelessBuilder.wrap_constant_literal(value)
        elif is_builtin_callable(value):
            return BuiltinVariable(value)
        elif is_allowed(value):
            if is_user_defined_allowed(value):
                self.tx.output.has_user_defined_allowed_in_graph = True
            return TorchVariable(value)
        elif isinstance(value, types.FunctionType):
            return UserFunctionVariable(value)
        elif isinstance(value, enum.Enum):
            return EnumVariable(value)
        elif isinstance(value, (type, abc.ABCMeta)):
            return UserDefinedClassVariable(value)
        elif isinstance(value, dict):
            return ConstDictVariable({k: self(tx, v) for k, v in value.items()}, dict, mutable_local=MutableLocal())
        elif isinstance(value, set):
            return SetVariable([self(tx, x) for x in value], mutable_local=MutableLocal())
        elif isinstance(value, (tuple, list)):
            cls = BaseListVariable.cls_for(type(value))
            return cls([self(tx, x) for x in value], mutable_local=MutableLocal())
        elif isinstance(value, types.MethodWrapperType):
            return MethodWrapperVariable(value)
        unimplemented(f'Unexpected type in sourceless builder {type(value)}')

    @staticmethod
    def wrap_constant_literal(value):
        assert ConstantVariable.is_literal(value)
        return ConstantVariable.create(value=value)