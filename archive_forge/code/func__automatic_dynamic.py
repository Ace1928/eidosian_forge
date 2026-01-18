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
def _automatic_dynamic(e, tx, source, static_shapes) -> SymbolicContext:
    name = source.name()
    prior_policy = tx.output.tracing_context.tensor_to_context.get(e, None)
    source_to_symint_node_cache = prior_policy.source_to_symint_node_cache if prior_policy else None
    if static_shapes:
        return StatefulSymbolicContext(dynamic_sizes=[DimDynamic.STATIC] * e.dim(), constraint_sizes=[None] * e.dim(), tensor_source=source, source_to_symint_node_cache=source_to_symint_node_cache)
    if any((isinstance(s, SymInt) for s in e.size())):
        return StatefulSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC if isinstance(s, SymInt) else DimDynamic.STATIC for s in e.size()], constraint_sizes=[None] * e.dim(), tensor_source=source, source_to_symint_node_cache=source_to_symint_node_cache)
    frame_state_entry = None
    if name not in tx.output.frame_state:
        frame_state_entry = FrameStateSizeEntry(None, None)
        frame_state_entry.size = list(e.size())
    else:
        frame_state_entry = tx.output.frame_state[name]
        if frame_state_entry.size is not None:
            if e.ndim != len(frame_state_entry.size):
                log.debug('automatic dynamic %s dim %s != %s', name, e.ndim, frame_state_entry.size)
                frame_state_entry.size = None
            else:
                for i, dim in enumerate(frame_state_entry.size):
                    if dim is not None and e.size()[i] != dim:
                        log.debug('automatic dynamic %s size(%s) %s != %s', name, i, e.size(i), dim)
                        frame_state_entry.size[i] = None
    t_id = id(e)
    dim2constraint = {}

    def update_dim2constraint(dim, constraint_range, debug_name):
        if dim in dim2constraint:
            from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
            old_constraint_range, old_debug_name = dim2constraint[dim]
            new_constraint_range = StrictMinMaxConstraint(vr=constraint_range.vr & old_constraint_range.vr, warn_only=False)
            if old_debug_name is not None:
                assert debug_name is None or debug_name == old_debug_name
                new_debug_name = old_debug_name
            else:
                new_debug_name = debug_name
            dim2constraint[dim] = (new_constraint_range, new_debug_name)
        else:
            dim2constraint[dim] = (constraint_range, debug_name)
    if tx.output.export_constraints:
        for constraint in tx.output.export_constraints:
            if constraint.t_id == t_id:
                update_dim2constraint(constraint.dim, constraint.constraint_range, constraint.debug_name)
            if constraint.shared is not None and constraint.shared.t_id == t_id:
                update_dim2constraint(constraint.shared.dim, constraint.constraint_range, constraint.debug_name)
    dynamic_dims = []
    constraint_dims = []
    for i in range(e.dim()):
        marked_dynamic = i in getattr(e, '_dynamo_dynamic_indices', set())
        marked_weak_dynamic = i in getattr(e, '_dynamo_weak_dynamic_indices', set())
        marked_static = i in getattr(e, '_dynamo_static_indices', set())
        automatic_dynamic = config.automatic_dynamic_shapes and (frame_state_entry.size is None or frame_state_entry.size[i] is None)
        if frame_state_entry.size and marked_dynamic:
            log.debug('automatic dynamic %s marked dynamic', name)
            frame_state_entry.size[i] = None
        constraint = dim2constraint.get(i)
        if constraint is None:
            if marked_dynamic and (not config.allow_ignore_mark_dynamic):
                constraint_dim = RelaxedUnspecConstraint(warn_only=False)
            elif not marked_static and automatic_dynamic:
                constraint_dim = RelaxedUnspecConstraint(warn_only=True)
            else:
                constraint_dim = None
        else:
            constraint_dim, debug_name = constraint
            if debug_name is not None:
                dim_name = f'{name}.size()[{i}]'
                tx.output.shape_env.source_name_to_debug_name[dim_name] = debug_name
        constraint_dims.append(constraint_dim)
        if constraint_dim is not None or marked_dynamic or marked_weak_dynamic:
            dynamic = DimDynamic.DYNAMIC
        elif static_shapes or config.assume_static_by_default or marked_static:
            dynamic = DimDynamic.STATIC
        else:
            dynamic = DimDynamic.DUCK
        dynamic_dims.append(dynamic)
    tx.output.frame_state[name] = frame_state_entry
    return StatefulSymbolicContext(dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims, tensor_source=source, source_to_symint_node_cache=source_to_symint_node_cache)