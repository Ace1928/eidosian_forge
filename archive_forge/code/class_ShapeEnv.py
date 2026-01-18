import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
class ShapeEnv:

    def __init__(self, *, should_record_events: Optional[bool]=None, tracked_fakes: Optional[List[Any]]=None, **kwargs) -> None:
        self._init(**kwargs)
        kwargs['should_record_events'] = False
        from torch.fx.experimental.validator import translation_validation_enabled
        self._translation_validation_enabled = translation_validation_enabled()
        self.should_record_events = should_record_events if should_record_events is not None else self._translation_validation_enabled and (not config.translation_validation_no_bisect)
        self.check_recorded_events = self.should_record_events and config.check_shape_env_recorded_events
        self.is_recording = not self.should_record_events
        self.tracked_fakes = tracked_fakes
        self.events: List[ShapeEnvEvent] = [ShapeEnvEvent(ShapeEnv, kwargs=kwargs)] if self.should_record_events else []

    def _init(self, *, allow_scalar_outputs=True, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, co_fields=None):
        self.allow_scalar_outputs = allow_scalar_outputs
        self.allow_dynamic_output_shape_ops = allow_dynamic_output_shape_ops
        self.guards: List[ShapeGuard] = []
        self.var_to_val: Dict[sympy.Symbol, sympy.Integer] = {}
        self.var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.source_name_to_debug_name: Dict[str, str] = {}
        self.runtime_var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.var_to_sources: Dict[sympy.Symbol, List[Source]] = {}
        self.var_to_stack: Dict[sympy.Symbol, CapturedTraceback] = {}
        self.var_to_guards: Dict[sympy.Symbol, Tuple[Optional[ShapeGuard], Optional[ShapeGuard]]] = {}
        self.replacements: Dict[sympy.Symbol, sympy.Expr] = {}
        self.divisible: Set[sympy.Expr] = set()
        self.val_to_var: Dict[int, sympy.Expr] = {}
        if specialize_zero_one:
            self.val_to_var = {0: sympy.Integer(0), 1: sympy.Integer(1)}
        self.unbacked_symfloat_counter = itertools.count()
        self.unbacked_symint_counter = itertools.count()
        self.deferred_runtime_asserts: Dict[sympy.Symbol, List[RuntimeAssert]] = {}
        self.num_deferred_runtime_asserts = 0
        self.assume_static_by_default = assume_static_by_default
        self.specialize_zero_one = specialize_zero_one
        self.duck_shape = duck_shape
        self.log = log
        self.log.info('create_env')
        self.frozen = False
        self.dim_constraints: Optional[DimConstraints] = None
        self.counter = collections.Counter()
        self.co_fields = co_fields if co_fields else {}
        self._prev_cache_key = self._get_key()
        self._version_counter = 0
        self.fx_node_cache: Dict[Tuple[Callable, Tuple[Any, ...]], torch.fx.Node] = {}
        self.source_to_symbol: Dict[str, sympy.Symbol] = {}
        from torch.fx.experimental.validator import translation_validation_enabled
        self._translation_validation_enabled = translation_validation_enabled()
        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import TranslationValidator
            self.validator = TranslationValidator()
            self.graph = torch.fx.Graph()
            self.graph.inserting_before(self.graph.output(None))
            self.name_to_node: Dict[str, torch.fx.Node] = {}

    def check_equal(self, other: 'ShapeEnv') -> None:
        non_state_variable_names = ('counter', 'log', 'var_to_stack', 'fx_node_cache', 'graph', 'validator', 'check_recorded_events', 'should_record_events', 'is_recording', 'tracked_fakes', 'events', 'source_name_to_debug_name', '_prev_cache_key', '_version_counter')

        def map_value(key: str, value: Any) -> Any:
            if key in ('unbacked_symfloat_counter', 'unbacked_symint_counter'):
                from copy import copy
                return next(copy(value))
            elif key == 'guards':
                return [g.expr for g in value]
            elif key == 'var_to_guards':
                return {s: (lb.expr if lb is not None else None, ub.expr if ub is not None else None) for s, (lb, ub) in value.items()}
            elif key == 'deferred_runtime_asserts':
                return {s: [ra.expr for ra in ras] for s, ras in value.items()}
            elif key == 'name_to_node':
                return set(value.keys())
            return value
        shape_env_check_state_equal(self, other, non_state_variable_names, map_value)

    def snapshot_tracked_fakes(self) -> Optional[List[Any]]:
        if self.tracked_fakes is None:
            return None
        from torch._dynamo.variables.builder import TrackedFake

        def maybe_transform_fake(fake: TrackedFake):
            inner_fake = fake.fake if isinstance(fake.fake, torch.SymInt) else FakeTensorMeta.from_fake(fake.fake)
            return TrackedFake(inner_fake, fake.source, fake.constraint_dims)
        return [maybe_transform_fake(fake) for fake in self.tracked_fakes]

    def inc_tracked_fakes_length(self) -> None:
        self.tracked_fakes_length += 1

    def set_tracked_fakes_length(self, i: int) -> None:
        self.tracked_fakes_length = i

    def last_event_index(self) -> int:
        return len(self.events) - 1

    @contextmanager
    def recording(self):
        self.is_recording = True
        try:
            yield
        finally:
            self.is_recording = False

    @record_shapeenv_event()
    def freeze(self):
        self.frozen = True

    def _create_symbol_for_source(self, source: Source) -> Optional[sympy.Symbol]:
        if not self._translation_validation_enabled:
            return None
        srcname = source.name()
        if source not in self.source_to_symbol:
            self.source_to_symbol[srcname] = sympy.Symbol(srcname, integer=True)
        return self.source_to_symbol[srcname]

    def _add_z3var(self, symbol: sympy.Symbol, type: Type) -> None:
        if self._translation_validation_enabled:
            self.validator.add_var(symbol, type)

    def _add_target_expr(self, expr) -> None:
        if self._translation_validation_enabled:
            self.validator.add_target_expr(expr)

    def _add_assertion(self, expr) -> None:
        if self._translation_validation_enabled:
            self.validator.add_assertion(expr)

    def _check_translation_validate(self) -> None:
        if self._translation_validation_enabled:
            self.validator.validate()

    @record_shapeenv_event()
    def create_fx_call_function(self, op: Callable, args: Tuple) -> Tuple[Optional[torch.fx.Node], bool]:
        node_key = (op, args)
        fresh = False
        if self._translation_validation_enabled and node_key not in self.fx_node_cache:
            from torch.fx.experimental.validator import z3op
            if any((a is None for a in args)):
                assert all((not isinstance(a, torch.fx.Node) for a in args))
                return (None, fresh)
            fresh = True
            lifted_op = z3op(op, self.validator)
            assert all((a is not None for a in args)), f'missing arg in FX graph ({op.__name__}): {args}'
            node = self.fx_node_cache[node_key] = self.graph.call_function(lifted_op, args)
            self.name_to_node[node.name] = node
        return (self.fx_node_cache.get(node_key, None), fresh)

    def create_fx_placeholder_and_z3var(self, symbol: sympy.Symbol, type: Type) -> Optional[torch.fx.Node]:
        if not self._translation_validation_enabled:
            return None
        node_key = (self.graph.placeholder, (symbol,))
        if node_key not in self.fx_node_cache:
            self._add_z3var(symbol, type)
            mangled_name = re.sub('[^a-zA-Z0-9]', '_', re.sub('[()]', '', symbol.name))
            node = self.fx_node_cache[node_key] = self.graph.placeholder(mangled_name)
            self.name_to_node[node.name] = node
            node.meta['symbol'] = symbol
        return self.fx_node_cache[node_key]

    def remove_fx_node(self, node: Optional[torch.fx.Node]) -> None:
        if self._translation_validation_enabled and node is not None:
            self.name_to_node.pop(node.name)
            self.graph.erase_node(node)

    def add_fx_node_metadata(self, node: torch.fx.Node) -> None:
        from torch._dynamo.utils import get_current_node
        if self.should_record_events:
            node.meta[SHAPEENV_EVENT_KEY] = self.last_event_index()
            node.meta[CURRENT_NODE_KEY] = get_current_node()

    def _suppress_guards_tls(self):
        return getattr(TLS, 'suppress_guards', False)

    @record_shapeenv_event()
    def suppress_guards_enter(self):
        TLS.suppress_guards = True

    @record_shapeenv_event()
    def suppress_guards_exit(self):
        TLS.suppress_guards = False

    @contextmanager
    def suppress_guards(self):
        self.suppress_guards_enter()
        try:
            yield
        finally:
            self.suppress_guards_exit()

    def _get_key(self):
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (len(self.replacements), len(self.divisible), self.num_deferred_runtime_asserts)

    def _update_version_counter(self):
        cur_key = self._get_key()
        if self._prev_cache_key != cur_key:
            self._prev_cache_key = cur_key
            self._version_counter += 1

    def _produce_dyn_sizes(self, ex_size: Sequence[int], source: Source, symbolic_context: SymbolicContext) -> List[sympy.Expr]:
        return self._produce_dyn_sizes_from_int_tuple(tuple(ex.size()), source, symbolic_context)

    def _produce_dyn_sizes_from_int_tuple(self, tensor_size: Tuple[int], source: Source, symbolic_context: SymbolicContext) -> List[sympy.Expr]:
        assert all((not is_symbolic(val) for val in tensor_size)), f'Expect size to be a plain tuple of ints but got {tensor_size}'
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        _assert_symbol_context(symbolic_context)
        dynamic_dims = symbolic_context.dynamic_sizes
        constraint_dims = symbolic_context.constraint_sizes
        size = []
        for i, val in enumerate(tensor_size):
            size.append(self.create_symbol(val, TensorPropertySource(source, TensorProperty.SIZE, i), dynamic_dims[i], constraint_dims[i]))
        return size

    def create_symbolic_sizes_strides_storage_offset(self, ex: torch.Tensor, source: Source, *, symbolic_context: Optional[SymbolicContext]=None):
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """
        assert not ex.is_nested

        def maybe_specialize_sym_int_with_hint(maybe_sym) -> int:
            assert isinstance(maybe_sym, (int, torch.SymInt))
            if is_symbolic(maybe_sym):
                assert maybe_sym.node.shape_env is not self, 'expect the symbol is created from an shape env other than current one.'
                return maybe_sym.node.require_hint()
            return maybe_sym
        ex_size = tuple((maybe_specialize_sym_int_with_hint(sz) for sz in ex.size()))
        ex_stride = tuple((maybe_specialize_sym_int_with_hint(sd) for sd in ex.stride()))
        ex_storage_offset = maybe_specialize_sym_int_with_hint(ex.storage_offset())
        return self._create_symbolic_sizes_strides_storage_offset(ex_size, ex_stride, ex_storage_offset, [_is_dim_dynamic(ex, i) for i in range(ex.dim())], source, symbolic_context=symbolic_context)

    @record_shapeenv_event()
    def _create_symbolic_sizes_strides_storage_offset(self, ex_size: Sequence[int], ex_stride: Sequence[int], ex_storage_offset: int, is_dim_dynamic: Sequence[bool], source: Source, *, symbolic_context: Optional[SymbolicContext]=None):
        dim = len(ex_size)
        if symbolic_context is None:
            constraint_dims = [None] * dim
            dynamic_dims = []
            for i in range(dim):
                if is_dim_dynamic[i]:
                    r = DimDynamic.DYNAMIC
                elif self.assume_static_by_default:
                    r = DimDynamic.STATIC
                else:
                    r = DimDynamic.DUCK
                dynamic_dims.append(r)
            dynamic_dims = [DimDynamic.DUCK] * dim
            symbolic_context = StatelessSymbolicContext(dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims)
        _assert_symbol_context(symbolic_context)
        constraint_dims = symbolic_context.constraint_sizes
        dynamic_dims = symbolic_context.dynamic_sizes
        dynamic_strides_offset = DimDynamic.STATIC if all((r == DimDynamic.STATIC for r in dynamic_dims)) else DimDynamic.DUCK
        assert len(dynamic_dims) == dim, f'{len(dynamic_dims)} != {dim}'
        assert len(constraint_dims) == dim
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        size: List[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(ex_size, source, symbolic_context)
        stride: List[Optional[sympy.Expr]] = [None] * len(size)
        for i, val in enumerate(ex_stride):
            if val in (0, 1):
                stride[i] = sympy.Integer(val)
        while any((x is None for x in stride)):
            candidates = {ex_size[i] * ex_stride[i]: size[i] * stride[i] for i in range(len(size)) if stride[i] is not None and ex_stride[i] >= 0}
            val_list = sorted([(ex_stride[i], i) for i in range(len(stride)) if stride[i] is None])
            for _, i in val_list:
                if stride[i] is None and ex_stride[i] in candidates:
                    stride[i] = candidates[ex_stride[i]]
                    candidates[ex_size[i] * ex_stride[i]] = size[i] * stride[i]
            if any((x is None for x in stride)):
                val, i = min([(ex_stride[i], i) for i in range(len(stride)) if stride[i] is None])
                stride[i] = self.create_symbol(val, TensorPropertySource(source, TensorProperty.STRIDE, i), dynamic_dim=dynamic_strides_offset, constraint_dim=None)
        assert all((x is not None for x in stride))
        sym_sizes = [self.create_symintnode(sym, hint=hint, source=TensorPropertySource(source, TensorProperty.SIZE, i), symbolic_context=symbolic_context) for i, (sym, hint) in enumerate(zip(size, ex_size))]
        sym_stride = []
        for i, stride_expr in enumerate(stride):
            assert stride_expr is not None
            sym_stride.append(self.create_symintnode(stride_expr, hint=ex_stride[i], source=TensorPropertySource(source, TensorProperty.STRIDE, i), symbolic_context=symbolic_context))
        sym_storage_offset = self.create_symintnode(self.create_symbol(ex_storage_offset, TensorPropertySource(source, TensorProperty.STORAGE_OFFSET), dynamic_dim=dynamic_strides_offset, constraint_dim=None), hint=ex_storage_offset, source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET), symbolic_context=symbolic_context)
        return (tuple(sym_sizes), tuple(sym_stride), sym_storage_offset)

    @record_shapeenv_event()
    def create_symintnode(self, sym: 'sympy.Expr', *, hint: Optional[int], source: Optional[Source]=None, symbolic_context: Optional[SymbolicContext]=None):
        source_name = source.name() if source else None
        if self._translation_validation_enabled and source is not None:
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None
            fx_node = self.create_fx_placeholder_and_z3var(symbol, int)
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None
        if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
            if source_name in symbolic_context.source_to_symint_node_cache:
                return symbolic_context.source_to_symint_node_cache[source_name]
        if isinstance(sym, sympy.Integer):
            if hint is not None:
                assert int(sym) == hint
            out = int(sym)
        else:
            out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
        if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
            symbolic_context.source_to_symint_node_cache[source_name] = out
        return out

    @record_shapeenv_event()
    def create_unspecified_symint_and_symbol(self, value, source, dynamic_dim):
        return self.create_symintnode(self.create_unspecified_symbol(value, source=source, dynamic_dim=dynamic_dim), hint=value, source=source)

    def create_symboolnode(self, sym: 'sympy.Expr'):
        return SymBool(SymNode(sym, self, bool, None))

    @record_shapeenv_event()
    def create_unbacked_symfloat(self):
        symbol: sympy.Symbol = sympy.Symbol(f'f{next(self.unbacked_symfloat_counter)}')
        self.counter['create_unbacked_symbol'] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        self.var_to_range[symbol] = ValueRanges.unknown()
        fx_node = self.create_fx_placeholder_and_z3var(symbol, float)
        return SymFloat(SymNode(symbol, self, float, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unbacked_symint(self):
        symbol: sympy.Symbol = sympy.Symbol(f'i{next(self.unbacked_symint_counter)}', integer=True)
        self.counter['create_unbacked_symbol'] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = self._default_unspecified_value_range()
        fx_node = self.create_fx_placeholder_and_z3var(symbol, int)
        fsummary, user_tb, maybe_user_loc = self._get_stack_summary()
        log.info('create_unbacked_symbol %s [%s, %s]%s (%s)', symbol, vr.lower, vr.upper, maybe_user_loc, format_frame(fsummary))
        return SymInt(SymNode(symbol, self, int, None, fx_node=fx_node))

    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        return str(symbol).startswith('i')

    @record_shapeenv_event()
    def create_unbacked_symbool(self):
        symbol: sympy.Symbol = sympy.Symbol(f'i{next(self.unbacked_symint_counter)}', integer=True)
        self.counter['create_unbacked_symbol'] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        self.var_to_range[symbol] = ValueRanges(0, 1)
        fx_node = self.create_fx_placeholder_and_z3var(symbol, bool)
        return SymBool(SymNode(sympy.Eq(symbol, 1), self, bool, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unspecified_symbol(self, val: Union[int, SymInt], source: Source, dynamic_dim: DimDynamic=DimDynamic.DUCK, constraint_dim: DimConstraint=None) -> 'sympy.Expr':
        return self.create_symbol(val, source, dynamic_dim, constraint_dim, positive=None, do_not_specialize_zero_one=True)

    @record_shapeenv_event()
    def create_symbol(self, val: int, source: Source, dynamic_dim: DimDynamic=DimDynamic.DUCK, constraint_dim: DimConstraint=None, positive: Optional[bool]=True, do_not_specialize_zero_one: bool=False) -> 'sympy.Expr':
        if do_not_specialize_zero_one:
            specialize_zero_one = False
        else:
            specialize_zero_one = self.specialize_zero_one
        assert isinstance(source, Source), f'{type(source)} {source}'
        assert not (positive and val < 0), f'positive set for negative value: {val}'
        if constraint_dim is not None:
            dynamic_dim = DimDynamic.DYNAMIC
        if dynamic_dim is DimDynamic.STATIC:
            return sympy.Integer(val)
        elif dynamic_dim is DimDynamic.DUCK:
            duck = self.duck_shape
        elif dynamic_dim is DimDynamic.DYNAMIC:
            duck = False
        else:
            raise AssertionError(f'unhandled dynamic_dim {dynamic_dim}')
        if val in (0, 1) and specialize_zero_one:
            r = self.val_to_var[val]
        elif not duck or val not in self.val_to_var:
            sympy_expr = sympy.Symbol(f's{len(self.var_to_val)}', positive=positive, integer=True)
            if isinstance(val, int):
                self.var_to_val[sympy_expr] = sympy.Integer(val)
            else:
                self.var_to_val[sympy_expr] = SingletonInt(val.node.singleton_int(), coeff=val.node.singleton_coeff())
            self.var_to_sources[sympy_expr] = []
            self._add_z3var(sympy_expr, int)
            if duck:
                self.val_to_var[val] = sympy_expr
            if isinstance(val, int):
                if positive:
                    self._add_assertion(sympy_expr > 1)
                    self.var_to_range[sympy_expr] = self._default_value_range()
                else:
                    self.var_to_range[sympy_expr] = self._default_unspecified_value_range()
                if isinstance(constraint_dim, StrictMinMaxConstraint):
                    assert not duck
                    self.var_to_range[sympy_expr] &= constraint_dim.vr
                vr = self.var_to_range[sympy_expr]
                if val not in vr:
                    raise ConstraintViolationError(f'{val} not in range [{vr.lower}, {vr.upper}]')
                self.runtime_var_to_range[sympy_expr] = vr
                range_str = f'[{vr.lower}, {vr.upper}]'
            else:
                range_str = ''
            r = sympy_expr
            self.log.info('create_symbol %s = %s for %s %s', sympy_expr, val, source.name(), range_str)
            self.counter['create_symbol'] += 1
        else:
            r = self.val_to_var[val]
            self.log.debug('create_symbol %s duck sized %s', r, source.name())
        if isinstance(r, sympy.Symbol):
            self.var_to_sources[r].append(source)
        return r

    def debug_name(self, source):
        src_name = source.name()
        return self.source_name_to_debug_name.get(src_name, src_name)

    def render_range_for_constraint_violation(self, source, c):
        if isinstance(c, StrictMinMaxConstraint):
            lower, upper = (c.vr.lower, c.vr.upper)
            default = self._default_value_range()
            if lower <= default.lower:
                lower = None
            if upper >= default.upper:
                upper = None
            c_render = f'{self.debug_name(source)} = {source.name()} in the specified range'
            if lower is not None and upper is not None:
                c_render += f' {lower} <= {self.debug_name(source)} <= {upper}'
            elif lower is None and upper is not None:
                c_render += f' {self.debug_name(source)} <= {upper}'
            elif lower is not None and upper is None:
                c_render += f' {lower} <= {self.debug_name(source)}'
            return c_render
        return c.render(source)

    def produce_guards(self, placeholders, sources, source_ref=lambda n: n.name(), *, constraint_inputs: Optional[InputList[Union[DimConstraint, Optional[DimList[DimConstraint]]]]]=None, equalities_inputs: Optional[Set[Tuple[Source, Source]]]=None, _simplified=False, ignore_static=True) -> List[str]:
        self.log.info('produce_guards')
        if self.check_recorded_events:
            shape_env = replay_shape_env_events(self.events)
            self.check_equal(shape_env)
        assert len(placeholders) == len(sources)
        Tensorlike = (torch.Tensor, FakeTensorMeta)
        if constraint_inputs is None:
            constraint_inputs = [[None] * t.dim() if isinstance(t, Tensorlike) else None for t in placeholders]
        else:
            assert len(constraint_inputs) == len(placeholders)
            for i, (t, constraint) in enumerate(zip(placeholders, constraint_inputs)):
                if isinstance(t, Tensorlike):
                    if constraint is None:
                        constraint_inputs[i] = [None] * t.dim()
                    else:
                        assert len(constraint) == t.dim()
                else:
                    assert isinstance(t, (SymInt, int))
                    assert not isinstance(constraint, list)
        from torch._dynamo.source import TensorPropertySource, TensorProperty, NegateSource
        input_guards = []
        symbol_to_source = collections.defaultdict(list)
        symbol_to_constraints = collections.defaultdict(set)
        constraint_violations: List[Tuple[bool, Callable[[], str]]] = []

        def record_constraint_violation(warn_only, debug_name, msg, hint=None):
            constraint_violations.append((warn_only, debug_name, lambda: f'{msg}{hint()}' if hint else msg))

        def is_dim(src):
            return isinstance(src, TensorPropertySource) and src.prop is TensorProperty.SIZE
        if equalities_inputs:
            source_index = {}
            for i, src in enumerate(sources):
                source_index[src.name()] = i

            def get_symbol(tensor_dim_src):
                fake = placeholders[source_index[tensor_dim_src.base.name()]]
                symint = fake.shape[tensor_dim_src.idx]
                assert isinstance(symint, torch.SymInt)
                return symint.node.expr
            for src1, src2 in equalities_inputs.source_pairs:
                s1, s2 = (get_symbol(src1), get_symbol(src2))
                concrete_val = self.evaluate_expr(sympy.Eq(s1, s2))
                if not concrete_val:
                    raise ConstraintViolationError(f'{src1.name()} = {self.var_to_val[s1]} is not equal to {src2.name()} = {self.var_to_val[s2]}')

        def track_symint(source, val, constraint=None):
            log.debug('track_symint %s %s %s', LazyString(source.name), val, constraint)
            assert not isinstance(val, SymInt) or is_symbolic(val)
            if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
                val = val.node.maybe_as_int()
            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                    if constraint is not None:
                        symbol_to_constraints[s].add(constraint)
                elif isinstance(-s, sympy.Symbol):
                    symbol_to_source[-s].append(NegateSource(source))
                else:
                    constraint_violated = False
                    if isinstance(constraint, StrictMinMaxConstraint):
                        sym_vrs = {x: self.var_to_range.get(x, None) for x in s.free_symbols}
                        if all((vr is not None for vr in sym_vrs.values())):
                            expr_vr = bound_sympy(s, sym_vrs)
                            if expr_vr != constraint.vr:
                                constraint_violated = True
                        else:
                            constraint_violated = True
                    elif isinstance(constraint, RelaxedUnspecConstraint):
                        if s.is_number:
                            i = int(s)
                            if i not in (0, 1):
                                constraint_violated = True
                        else:
                            constraint_violated = True
                    if constraint_violated:

                        def hint(s):
                            sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(s)
                            return f'{sexpr}.'
                        var_with_range = self.render_range_for_constraint_violation(source, constraint)
                        msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be equal to '
                        record_constraint_violation(constraint.warn_only, self.debug_name(source), msg, hint=functools.partial(hint, s))
                input_guards.append((source, s))
            else:
                s = sympy.Integer(val)
                input_guards.append((source, s))
                constraint_violated = False
                if isinstance(constraint, StrictMinMaxConstraint):
                    constraint_violated = True
                elif isinstance(constraint, RelaxedUnspecConstraint):
                    if val not in (0, 1):
                        constraint_violated = True
                if constraint_violated:
                    var_with_range = self.render_range_for_constraint_violation(source, constraint)
                    msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be a constant ({val}).'
                    record_constraint_violation(constraint.warn_only, self.debug_name(source), msg)
        for t, source, constraint in zip(placeholders, sources, constraint_inputs):
            if isinstance(source, str):
                from torch._dynamo.source import LocalSource
                source = LocalSource(source)
            assert isinstance(source, Source)
            if t is None:
                continue
            if isinstance(t, (SymInt, int)):
                track_symint(source, t)
                continue
            assert isinstance(t, Tensorlike)
            sources_and_tensors = [(source, t)]
            if is_traceable_wrapper_subclass(t):
                attrs, _ = t.__tensor_flatten__()
                from torch._dynamo.source import AttrSource
                inner_sources_and_tensors = [(AttrSource(source, attr), getattr(t, attr)) for attr in attrs]
                if t.is_nested:
                    sources_and_tensors.extend(inner_sources_and_tensors)
                else:
                    sources_and_tensors = inner_sources_and_tensors
            for src, curr_t in sources_and_tensors:
                for i, ss in enumerate(curr_t.size()):
                    property_source = TensorPropertySource(src, TensorProperty.SIZE, i)
                    track_symint(property_source, ss, constraint[i])
                if not t.is_nested:
                    for i, ss in enumerate(curr_t.stride()):
                        track_symint(TensorPropertySource(src, TensorProperty.STRIDE, i), ss)
                    track_symint(TensorPropertySource(src, TensorProperty.STORAGE_OFFSET), curr_t.storage_offset())
        exprs = []
        self.dim_constraints = DimConstraints(symbol_to_source, self.var_to_val, set(symbol_to_constraints.keys()), self.source_name_to_debug_name)
        if not _simplified:
            for source, expr in input_guards:
                if self._translation_validation_enabled:
                    srcname = source.name()
                    if srcname in self.source_to_symbol:
                        self._add_target_expr(sympy.Eq(self.source_to_symbol[srcname], expr))
                if isinstance(expr, sympy.Symbol) and symbol_to_source.get(expr) and (source == symbol_to_source[expr][0]):
                    continue
                if ignore_static and isinstance(source, TensorPropertySource):
                    if expr.is_number:
                        self.log.debug('Skipping guard %s', f'{source_ref(source)} == {expr}')
                        continue
                if is_dim(source):
                    self.dim_constraints.add_equality(source, expr)
                sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(f'{source_ref(source)} == {sexpr}')
                if isinstance(expr, sympy.Symbol) and expr in symbol_to_constraints and isinstance(source, TensorPropertySource) and (source.prop is TensorProperty.SIZE) and equalities_inputs and (not equalities_inputs.is_equal(source, symbol_to_source[expr][0])):
                    msg = f'The values of {self.debug_name(source)} = {source.name()} and {self.debug_name(symbol_to_source[expr][0])} = {symbol_to_source[expr][0].name()} must always be equal.'
                    record_constraint_violation(equalities_inputs.warn_only, self.debug_name(source), msg)
        issued = set()

        def issue_guard(guard: ShapeGuard) -> None:
            expr = self.simplify(guard.expr)
            if expr in issued:
                return
            issued.add(expr)
            try:
                is_trivial = False
                if any((is_dim(source) for s in expr.free_symbols for source in symbol_to_source[s])):
                    is_trivial = self.dim_constraints.add(expr)
                guard_expr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(guard_expr)
                self._add_target_expr(expr)
                if not is_trivial and len(expr.free_symbols) == 1:
                    symbol = next(iter(expr.free_symbols))
                    source = symbol_to_source[symbol][0]
                    constraints = symbol_to_constraints[symbol]
                    for c in constraints:
                        if isinstance(c, StrictMinMaxConstraint):
                            var_with_range = self.render_range_for_constraint_violation(source, c)
                            msg = f'Not all values of {var_with_range} satisfy the generated guard {guard_expr}.'
                            record_constraint_violation(c.warn_only, self.debug_name(source), msg)
                        elif isinstance(c, RelaxedUnspecConstraint):
                            pass
                        else:
                            raise AssertionError(f'unrecognized constraint {c}')
            except Exception:
                self.log.warning('Failing guard allocated at: \n%s', ''.join(guard.stack.format()))
                raise
        for guard in self.guards:
            if self._maybe_evaluate_static(guard.expr) is not None:
                continue
            issue_guard(guard)
        for symbol, guards in self.var_to_guards.items():
            if symbol not in symbol_to_source:
                continue
            for guard in guards:
                if guard is not None:
                    issue_guard(guard)
        if not _simplified:
            for symbol, sources in symbol_to_source.items():
                r = self.runtime_var_to_range.get(symbol)
                if r is None:
                    if symbol not in self.var_to_range:
                        continue
                    r = self.var_to_range[symbol]
                assert sources
                assert symbol.is_integer
                g_lower, g_upper = self.var_to_guards.get(symbol, (None, None))
                bounds = []
                if r.lower != -sympy.oo and g_lower is None:
                    if any((is_dim(source) for source in sources)):
                        self.dim_constraints.add(sympy.Ge(symbol, r.lower))
                    bounds.append(str(r.lower))
                bounds.append(source_ref(sources[0]))
                if r.upper != sympy.oo and r.upper < sys.maxsize - 1 and (g_upper is None):
                    if any((is_dim(source) for source in sources)):
                        self.dim_constraints.add(sympy.Le(symbol, r.upper))
                    bounds.append(str(r.upper))
                if len(bounds) > 1:
                    exprs.append(' <= '.join(bounds))
        if constraint_violations:
            warn_msgs = []
            error_msgs = []
            debug_names = set()
            for warn_only, debug_name, msg in constraint_violations:
                if warn_only:
                    msg = f'  {len(warn_msgs) + 1}. {msg()}'
                    warn_msgs.append(msg)
                else:
                    msg = f'  - {msg()}'
                    error_msgs.append(msg)
                    debug_names.add(debug_name)
            if len(error_msgs) > 0:
                debug_names = ', '.join(debug_names)
                err = '\n'.join(error_msgs)
                raise ConstraintViolationError(f'Constraints violated ({debug_names})! For more information, run with TORCH_LOGS=dynamic.\n{err}')
            elif len(warn_msgs) > 0:
                log.debug('%s Warning only constraints violated', len(warn_msgs))
        signpost_event('dynamic', 'produce_guards', {**self.co_fields, **self.counter, 'num_guards': len(exprs), 'free_symbols': sum((1 for v in symbol_to_source.values() if v))})
        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import PopulateValidator
            for ras in self.deferred_runtime_asserts.values():
                for ra in ras:
                    self._add_target_expr(ra.expr)
            for sym, vr in self.var_to_range.items():
                if vr.lower != -sympy.oo:
                    self._add_target_expr(sympy.Le(vr.lower, sym))
                if vr.upper != sympy.oo:
                    self._add_target_expr(sympy.Le(sym, vr.upper))
            with fx_traceback.preserve_node_meta():
                PopulateValidator(self.graph, self.validator).run()
        self._check_translation_validate()
        return exprs

    def produce_guards_expression(self, placeholders, ignore_static=True):
        """
        Expected to be used with evaluate_guards_expression(). Produces the guards
        for the given placeholders and returns a string expression to be evaluated
        by evaluate_guards_expression given concrete values for the placeholders.
        """
        from torch._dynamo.source import LocalSource
        arg_names = [f't{i}' for i in range(len(placeholders))]
        guards = self.produce_guards(placeholders, [LocalSource(a) for a in arg_names], ignore_static=ignore_static)
        if guards:
            return ' and '.join(guards)
        return None

    def evaluate_guards_expression(self, code, args):
        """
        Expected to be used with produce_guards_expression(). Evaluates an expression
        generated by produce_guards_expression for the given concrete args.
        """
        arg_names = [f't{i}' for i in range(len(args))]
        return eval(code, SYMPY_INTERP, {'L': dict(zip(arg_names, args))})

    def evaluate_guards_for_args(self, placeholders, args, *, ignore_static=True):
        code = self.produce_guards_expression(placeholders, ignore_static=ignore_static)
        if code:
            return self.evaluate_guards_expression(code, args)
        return True

    def bind_symbols(self, placeholders, args):
        bindings: Dict[sympy.Symbol, int] = {}

        def bind_symint(arg, val):
            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    if s in bindings:
                        assert bindings[s] == arg, f'{bindings[s]} != {arg}'
                    else:
                        bindings[s] = arg
                elif isinstance(-s, sympy.Symbol):
                    if -s in bindings:
                        assert bindings[-s] == -arg, f'{bindings[-s]} != {-arg}'
                    else:
                        bindings[-s] = -arg
        for t, arg in zip(placeholders, args):
            if t is None:
                continue
            if isinstance(t, SymInt):
                bind_symint(arg, t)
                continue
            assert isinstance(t, torch.Tensor)
            for i, s in enumerate(t.size()):
                bind_symint(arg.size(i), s)
            for i, s in enumerate(t.stride()):
                bind_symint(arg.stride(i), s)
            bind_symint(arg.storage_offset(), t.storage_offset())
        return bindings

    def get_nontrivial_guards(self):
        return [self.simplify(guard.expr) for guard in self.guards if self._maybe_evaluate_static(guard.expr) is None]

    def format_guards(self, verbose=False):

        def format_tb(tb):
            if not verbose:
                return ''
            return f'\n   Guarded at:\n{''.join(('   ' + l for l in tb.format()))}'
        return '\n'.join((f' - {guard.expr}{format_tb(guard.stack)}' for guard in self.guards))

    def get_shape_groups(self):
        shape_groups = collections.defaultdict(list)
        for k, v in self.replacements.items():
            shape_groups[v].append(k)
        return shape_groups

    @_lru_cache
    def _maybe_evaluate_static(self, expr: 'sympy.Expr', *, unbacked_only: bool=False, compute_hint: bool=False, expect_rational=True) -> 'Optional[sympy.Expr]':
        """
        Tries to evaluate expr without introducing guards

        If unbacked_only == True, then we only do substitutions on
        unbacked SymInts (leaving regular hinted integers alone).  This could
        result in an expression that still contains backed SymInts, which you
        could then potentially guard on.

        Use compute_hint == True if you are trying to compute a non-binding
        hint for the particular hint values of backed SymInts, e.g., if
        s0 happens to be 3 this run, compute_hint will subsitute s0 with 3.
        """
        expr = self.simplify(expr)
        if compute_hint:
            expr = expr.xreplace(self.var_to_val)
        expr = canonicalize_bool_expr(expr)
        symbols = list(expr.free_symbols)
        for s in symbols:
            if s in self.var_to_val:
                continue
            subst = {}
            for ra in self.deferred_runtime_asserts.get(s, ()):
                if compute_hint:
                    e = canonicalize_bool_expr(ra.expr.xreplace(self.var_to_val))
                else:
                    e = ra.expr
                subst[e] = sympy.true
                subst[canonicalize_bool_expr(sympy.Not(e))] = sympy.false
                if isinstance(e, sympy.Eq):
                    subst[sympy.Le(e.lhs, e.rhs)] = sympy.true
                    subst[sympy.Le(-e.lhs, -e.rhs)] = sympy.true
                    subst[sympy.Lt(e.lhs, e.rhs)] = sympy.false
                    subst[sympy.Lt(-e.lhs, -e.rhs)] = sympy.false
            expr = expr.subs(subst)
        new_shape_env = {}
        new_range_env = {}
        for idx, k in enumerate(symbols):
            if isinstance(self.var_to_val.get(k, None), SingletonInt):
                continue
            vr = self.var_to_range[k]
            if vr.lower < (-sys.maxsize - 1) // 2 or (unbacked_only and k in self.var_to_val):
                new_range_env[k] = vr
                continue
            s = sympy.Symbol(f'shape_{idx}', positive=True, integer=True)
            offset = vr.lower - 1
            new_shape_env[k] = s + offset
            new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

        def replace(expr, repl):
            return expr.xreplace(repl)
        try:
            new_expr = replace(expr, new_shape_env)
        except RecursionError:
            log.warning('RecursionError in sympy.xreplace(%s, %s)', expr, new_shape_env)
            self.counter['sympy_recursion_error'] += 1
            return None
        floor_div_replace = {}
        for atom in new_expr.atoms(FloorDiv):
            floor_div_replace[atom] = sympy.floor(atom.args[0] / atom.args[1])
        new_expr = safe_expand(new_expr.xreplace(floor_div_replace))
        if new_expr.is_number:
            return new_expr
        out = bound_sympy(new_expr, new_range_env)
        if expect_rational:
            _assert_bound_is_rational(new_expr, out)
            if out.is_singleton():
                return out.lower
        return new_expr if unbacked_only else None

    @_lru_cache
    def replace(self, expr: 'sympy.Expr') -> 'sympy.Expr':
        replacements = {s: self._find(cast(sympy.Symbol, s)) for s in expr.free_symbols}
        return safe_expand(expr.xreplace(replacements))

    @_lru_cache
    def _update_divisible(self):
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if not res.is_number:
                new_divisible.add(k)
        self.divisible = new_divisible
        self._update_version_counter()

    @_lru_cache
    def simplify(self, expr: 'sympy.Expr') -> 'sympy.Expr':
        expr = self.replace(expr)
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                base, divisor = atom.args
                if isinstance(divisor, FloorDiv):
                    base1, divisor1 = divisor.args
                    if self.replace(Mod(base, divisor)) in self.divisible and base == base1 and (self.replace(Mod(base1, divisor1)) in self.divisible):
                        div_replacements[atom] = divisor1
            expr = expr.xreplace(div_replacements)
            expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                base, divisor = fd.args
                if self.replace(Mod(base, divisor)) in self.divisible:
                    div_replacements[fd] = base / divisor
            new_expr = expr.xreplace(div_replacements)
            new_expr = safe_expand(new_expr)
            new_pows = new_expr.atoms(sympy.Pow)
            new_rationals = new_expr.atoms(sympy.Rational).difference(new_expr.atoms(sympy.Integer))
            if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                expr = new_expr
        return expr

    @lru_cache(256)
    def size_hint(self, expr: 'sympy.Expr', *, allow_none=False):
        """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        if not result_expr.is_number:
            r = self._maybe_evaluate_static(result_expr, compute_hint=True)
            if r is not None:
                return r
            if allow_none:
                return None
            raise self._make_data_dependent_error(result_expr, expr)
        return result_expr

    @lru_cache(256)
    def has_hint(self, expr: 'sympy.Expr'):
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        return result_expr.is_number or self._maybe_evaluate_static(result_expr) is not None

    def _make_data_dependent_error(self, expr, unhinted_expr):
        for s in expr.free_symbols:
            stacktrace = ''.join(self.var_to_stack[s].format())
            self.log.debug("Data dependent variable '%s' allocated at:\n%s", s, stacktrace)
        return GuardOnDataDependentSymNode(f"It appears that you're trying to get a value out of symbolic int/float whose value is data-dependent (and thus we do not know the true value.)  The expression we were trying to evaluate is {expr} (unhinted: {unhinted_expr}).  Scroll up to see where each of these data-dependent accesses originally occurred.")

    def _set_replacement(self, a: 'sympy.Symbol', expr: 'sympy.Expr') -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = expr`.
        """
        if config.print_specializations and isinstance(expr, (sympy.Integer, sympy.Float)):
            if a not in self.replacements or expr != self.replacements[a]:
                self.log.warning('Specializing %s to %s', self.var_to_sources[a][0].name(), expr)
                self.log.debug('SPECIALIZATION', stack_info=True)
        log.info('set_replacement %s = %s', a, expr)
        self.replacements[a] = expr
        self._update_version_counter()
        self._add_target_expr(sympy.Eq(a, expr))

    def _add_divisible(self, expr: 'sympy.Expr'):
        self.divisible.add(expr)
        self._update_version_counter()

    @_lru_cache
    @record_shapeenv_event()
    def _find(self, a: 'sympy.Symbol') -> 'sympy.Expr':
        """
        Implements a DSU-like algorithm to find the variable that represents a
        Also handles transitive non-identity replacements.

        a: b + c
        c: d
        """
        if a not in self.replacements:
            return a
        res = self.replacements[a]
        cur_replace = {s: self._find(s) for s in res.free_symbols}
        self._set_replacement(a, self.replacements[a].xreplace(cur_replace))
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_eq(self, expr: Union['sympy.Eq', 'sympy.Ne'], concrete_bool: bool) -> None:
        """
        Evaluates the result of an eq call. If true, uses information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
        assert type(concrete_bool) is bool
        if isinstance(expr, sympy.Eq):
            if not concrete_bool:
                return
        elif isinstance(expr, sympy.Ne):
            if concrete_bool:
                return
        free = list(expr.free_symbols)
        assert len(free) > 0, f'The expression should not be static by this point: {expr}'
        if len(free) > 5:
            return
        free = sorted(free, key=lambda x: (self.size_hint(x, allow_none=True) or sys.maxsize, x.name), reverse=True)
        lhs = expr.lhs
        rhs = expr.rhs
        if not expr.has(Mod):
            try:
                floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
                if len(floor_div_atoms) > 0 and any((a.divisor != 1 for a in floor_div_atoms)):
                    raise NotImplementedError
                if isinstance(lhs, sympy.Symbol) and free_unbacked_symbols(lhs):
                    self._set_replacement(lhs, self._find(rhs))
                elif isinstance(rhs, sympy.Symbol) and free_unbacked_symbols(rhs):
                    self._set_replacement(rhs, self._find(lhs))
                else:
                    r = try_solve(expr, free[0], floordiv_inequality=False)
                    if r is not None and all((t.is_integer for t in sympy.preorder_traversal(r[1]))):
                        new_var = self._find(r[1])
                        ok = False
                        if self.is_unbacked_symint(free[0]):
                            ok = len(free_unbacked_symbols(new_var)) <= 1
                        else:
                            ok = len(free_unbacked_symbols(new_var)) == 0
                        if ok:
                            self._set_replacement(cast(sympy.Symbol, free[0]), new_var)
            except NotImplementedError:
                pass
        if expr.has(Mod):
            mod_expr = next(iter(expr.atoms(Mod)))
            try:
                r = try_solve(expr, mod_expr, floordiv_inequality=False)
                if r is not None and r[1] == 0:
                    self._add_divisible(mod_expr)
                    p, q = mod_expr.args
                    if isinstance(q, sympy.Number) and isinstance(p, sympy.Mul) and (len(p.args) == 2):
                        c, i0 = p.args
                        if isinstance(c, sympy.Number) and isinstance(i0, sympy.Symbol) and self.is_unbacked_symint(i0):
                            d = q / sympy.gcd(q, c)
                            i1 = self.create_unbacked_symint().node.expr
                            self.var_to_range[i1] = SymPyValueRangeAnalysis.truediv(self.var_to_range[i0], ValueRanges.wrap(d))
                            self.runtime_var_to_range[i1] = SymPyValueRangeAnalysis.truediv(self.runtime_var_to_range[i0], ValueRanges.wrap(d))
                            self._set_replacement(i0, d * i1)
            except NotImplementedError:
                pass
        return

    def _default_value_range(self) -> ValueRanges:
        lower = 2 if self.specialize_zero_one else 0
        return ValueRanges(lower, sys.maxsize - 1)

    def _default_unspecified_value_range(self) -> ValueRanges:
        return ValueRanges(-sys.maxsize - 1, sys.maxsize)

    @_lru_cache
    def _simplify_floor_div(self, expr):
        floor_divs = tuple(expr.atoms(FloorDiv))
        for fd in reversed(floor_divs):
            base, divisor = fd.args
            mod_expr = Mod(base, divisor)
            eq_expr = sympy.Eq(mod_expr, 0)
            self.evaluate_expr(eq_expr)
        return self.simplify(expr)

    def _check_frozen(self, expr, concrete_val):
        if self.frozen:
            self.counter['ignored_backward_guard'] += 1
            signpost_event('dynamic', 'evaluate_expr_frozen', {**self.co_fields, 'ignored_guard': f'{expr} == {concrete_val}', 'version': 2})
            log.warning('Ignored guard %s == %s, this could result in accuracy problems', expr, concrete_val)

    def _get_stack_summary(self):
        fsummary = None
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_filename not in uninteresting_files():
                    fsummary = traceback.FrameSummary(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                    break
                frame = frame.f_back
        finally:
            del frame
        maybe_user_loc = ''
        user_tb = TracingContext.extract_stack()
        if user_tb:
            maybe_user_loc = ' at ' + format_frame(user_tb[-1])
        return (fsummary, user_tb, maybe_user_loc)

    def _log_guard(self, prefix: str, g):
        if self.log.isEnabledFor(logging.INFO):
            fsummary, user_tb, maybe_user_loc = self._get_stack_summary()
            is_debug = False
            maybe_extra_debug = ''
            if is_debug and user_tb:
                maybe_extra_debug = '\nUser Stack (most recent call last):\n' + '  (snipped, see stack below for prefix)\n' + ''.join(traceback.format_list(user_tb))
            self.log.info('%s %s [guard added]%s (%s)%s', prefix, g, maybe_user_loc, format_frame(fsummary), maybe_extra_debug, stack_info=is_debug)

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def evaluate_expr(self, orig_expr: 'sympy.Expr', hint=None, fx_node=None, expect_rational=True):
        """
        Given an expression, evaluates it, adding guards if necessary
        """
        if hint is None:
            concrete_val = self.size_hint(orig_expr)
        else:
            concrete_val = sympy.sympify(hint)
        node = None
        fresh = False
        if self._translation_validation_enabled and fx_node is not None and (not self._suppress_guards_tls()):
            if concrete_val is sympy.true:
                node, fresh = self.create_fx_call_function(torch._assert, (fx_node,))
            elif concrete_val is sympy.false:
                neg, _ = self.create_fx_call_function(operator.not_, (fx_node,))
                node, fresh = self.create_fx_call_function(torch._assert, (neg,))
            else:
                eql, _ = self.create_fx_call_function(operator.eq, (fx_node, concrete_val))
                node, fresh = self.create_fx_call_function(torch._assert, (eql,))
            assert node is not None
            if fresh:
                self.add_fx_node_metadata(node)
        guard = None
        tb = None
        try:
            if orig_expr.is_number:
                self.log.debug('eval %s [trivial]', orig_expr)
                if isinstance(hint, (int, bool)):
                    assert orig_expr == hint, f'{orig_expr} != {hint}'
                return orig_expr
            expr = orig_expr
            static_expr = self._maybe_evaluate_static(expr, expect_rational=expect_rational)
            if static_expr is not None:
                self.log.debug('eval %s == %s [statically known]', orig_expr, static_expr)
                if isinstance(hint, (int, bool)):
                    assert static_expr == hint, f'{static_expr} != {hint}'
                return static_expr
            if not expr.free_symbols <= self.var_to_val.keys():
                new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
                if not new_expr.free_symbols <= self.var_to_val.keys():
                    raise self._make_data_dependent_error(expr.xreplace(self.var_to_val), expr)
                expr = new_expr
            self._check_frozen(expr, concrete_val)
            if config.inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY and isinstance(hint, bool) and isinstance(expr, (sympy.Eq, sympy.Ne)):
                expr = sympy.Not(expr)
            if isinstance(expr, (sympy.Eq, sympy.Ne)):
                self._maybe_guard_eq(expr, bool(concrete_val))
            elif isinstance(concrete_val, sympy.Integer):
                self._maybe_guard_eq(sympy.Eq(expr, concrete_val), True)
            if concrete_val is sympy.true:
                g = expr
            elif concrete_val is sympy.false:
                g = sympy.Not(expr)
            else:
                g = sympy.Eq(expr, concrete_val)
            if not self._suppress_guards_tls():
                stack = CapturedTraceback.extract(skip=1)
                guard = ShapeGuard(g, stack)
                self.guards.append(guard)
        except Exception:
            if fresh:
                self.remove_fx_node(node)
            raise
        else:
            if not self._suppress_guards_tls():
                assert guard is not None
                self.refine_ranges(guard)
                self._log_guard('eval', g)
            else:
                self.log.debug('eval %s [guard suppressed]', g)
        return concrete_val

    def cleanup(self):
        for g in self.guards:
            g.stack.cleanup()
        for s in self.var_to_stack.values():
            s.cleanup()
        for ras in self.deferred_runtime_asserts.values():
            for ra in ras:
                ra.stack.cleanup()

    @record_shapeenv_event(save_tracked_fakes=True)
    def defer_runtime_assert(self, orig_expr: 'sympy.Expr', msg, fx_node=None):
        expr = orig_expr
        static_expr = self._maybe_evaluate_static(expr)
        if static_expr is not None:
            self.log.debug('runtime_assert %s == %s [statically known]', orig_expr, static_expr)
            return static_expr
        new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
        if new_expr.free_symbols <= self.var_to_val.keys():
            return self.evaluate_expr(new_expr, fx_node=fx_node)
        if self._translation_validation_enabled and fx_node is not None and (not self._suppress_guards_tls()):
            node, fresh = self.create_fx_call_function(torch._assert, (fx_node,))
            assert node is not None
            if fresh:
                self.add_fx_node_metadata(node)
        self._check_frozen(expr, sympy.true)
        if isinstance(expr, sympy.Eq):
            self._maybe_guard_eq(expr, True)
        if not self._suppress_guards_tls():
            expr = canonicalize_bool_expr(expr)
            stack = CapturedTraceback.extract(skip=1)
            ra = RuntimeAssert(expr, msg, stack)
            cands = sorted([s for s in expr.free_symbols if s.name.startswith('i')], key=lambda s: int(s.name[1:]))
            self.deferred_runtime_asserts.setdefault(cands[-1], []).append(ra)
            self.num_deferred_runtime_asserts += 1
            self._update_version_counter()
            self._log_guard('runtime_assert', expr)
        else:
            self.log.debug('runtime_assert %s [guard suppressed]', expr)
        return True

    def refine_ranges(self, guard: ShapeGuard) -> None:
        expr = self.simplify(guard.expr)
        for symbol in expr.free_symbols:
            assert isinstance(symbol, sympy.Symbol)
            if isinstance(self.var_to_val.get(symbol, None), SingletonInt):
                continue
            r = try_solve(expr, symbol)
            if r is None or not (symbol.is_integer and r[1].is_integer):
                continue
            r_expr, rhs = r
            vr = self.var_to_range[symbol]
            lower, upper = (vr.lower, vr.upper)
            rhs_vr = bound_sympy(rhs, self.var_to_range)
            _assert_bound_is_rational(rhs, rhs_vr)
            lower_guard, upper_guard = self.var_to_guards.get(symbol, (None, None))
            if lower < rhs_vr.lower and isinstance(r_expr, (sympy.Eq, sympy.Ge, sympy.Gt)):
                lower = rhs_vr.lower + int(isinstance(r_expr, sympy.Gt))
                lower_guard = guard
            if upper > rhs_vr.upper and isinstance(r_expr, (sympy.Eq, sympy.Le, sympy.Lt)):
                upper = rhs_vr.upper - int(isinstance(r_expr, sympy.Lt))
                upper_guard = guard
            if vr == ValueRanges(lower, upper):
                continue
            self.var_to_range[symbol] = ValueRanges(lower, upper)
            self.var_to_guards[symbol] = (lower_guard, upper_guard)
            self._maybe_evaluate_static.cache_clear()