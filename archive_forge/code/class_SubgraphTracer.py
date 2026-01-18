import collections
import contextlib
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import sympy
import torch._guards
import torch._logging
import torch.nn
import torch.utils._pytree as pytree
from torch import fx
from torch._guards import (
from torch._utils_internal import signpost_event
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import free_symbols, is_symbolic, ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.reference import PythonReferenceAnalysis
from torch.utils.weak import WeakTensorKeyDictionary
from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .exc import (
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import (
from .utils import (
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
class SubgraphTracer(fx.Tracer):
    """
    Holds an FX graph that is being traced. OutputGraph owns a SubgraphTracer
    and the separation of responsibilities is that SubgraphTracer is
    responsible for building the graph while OutputGraph is responsible for
    compiling and executing the graph.
    """

    def __init__(self, output_graph, parent=None, export_root=False, source_target=None):
        super().__init__()
        self.output_graph = weakref.proxy(output_graph)
        self.graph = torch.fx.Graph()
        if export_root:
            assert parent is None
        self.export_root = export_root
        self.input_name_to_proxy: Dict[str, fx.Proxy] = {}
        self.real_value_cache: Dict[fx.Node, torch.Tensor] = {}
        self.parent = parent
        self.lifted_freevars = {}
        self.prev_inst = None
        self._cur_code = None
        self._orig_gm_meta = None
        self._orig_gm_lineno_map = None
        self._orig_gm_firstlineno = None
        if self.parent is None:
            self.source_fn_stack = []
        else:
            self.source_fn_stack = self.parent.source_fn_stack + [(self.graph._target_to_str(source_target), source_target)]

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        if self.parent is not None:
            flat_args, tree_spec = pytree.tree_flatten((args, kwargs))
            new_flat_args = []
            for arg in flat_args:
                maybe_new_arg = self.maybe_lift_tracked_freevar_to_input(arg)
                new_flat_args.append(maybe_new_arg)
            args, kwargs = pytree.tree_unflatten(new_flat_args, tree_spec)
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        tx = self.output_graph.current_tx
        if sys.version_info >= (3, 11) and kind in ('call_function', 'call_method', 'call_module'):
            cur_inst = tx.current_instruction
            if cur_inst is not self.prev_inst and cur_inst.positions is not None and (cur_inst.positions.lineno is not None):
                tx_code = tx.f_code
                header = tx.get_line_of_code_header(lineno=cur_inst.positions.lineno)

                def get_trace_call_log_str():
                    line = get_instruction_source_311(tx_code, cur_inst).rstrip()
                    return f'TRACE FX call {rv.node.name} from {header}\n{line}'
                trace_call_log.debug('%s', LazyString(get_trace_call_log_str))
                self.prev_inst = cur_inst
        is_retracing = False
        if tx.f_code is not self._cur_code:
            orig_graphmodule_maybe = code_context.get_context(tx.f_code).get('orig_graphmodule', None)
            if isinstance(orig_graphmodule_maybe, torch.fx.GraphModule):
                is_retracing = True
                self._orig_gm_meta = [nd.meta for nd in orig_graphmodule_maybe.graph.nodes]
                self._orig_gm_lineno_map = orig_graphmodule_maybe._lineno_map
                self._orig_gm_firstlineno = orig_graphmodule_maybe.forward.__code__.co_firstlineno
            else:
                self._orig_gm_meta = None
                self._orig_gm_lineno_map = None
                self._orig_gm_firstlineno = None
        nn_module_stack = tx.nn_module_stack
        if nn_module_stack:
            rv.node.meta['nn_module_stack'] = nn_module_stack.copy()
        if kind in {'call_function', 'call_method'}:
            rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, target)]
        elif kind == 'call_module':
            if self.parent is not None:
                unimplemented('Invoking an nn.Module inside HigherOrderOperator')
            rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, rv.node.meta['nn_module_stack'][target][1])]
        if self._orig_gm_meta and self._orig_gm_lineno_map and self._orig_gm_firstlineno:
            lineno = tx.current_instruction.starts_line
            node_idx = None
            if lineno is not None:
                node_idx = self._orig_gm_lineno_map.get(lineno - self._orig_gm_firstlineno, None)
            if node_idx is not None:
                meta = self._orig_gm_meta[node_idx]
                for field in fx.proxy._COPY_META_FIELDS:
                    if field in meta:
                        rv.node.meta[field] = meta[field]
                if 'stack_trace' in meta:
                    rv.node.meta['stack_trace'] = meta['stack_trace']
        if not is_retracing:
            if 'nn_module_stack' not in rv.node.meta:
                nn_module_stack = tx.nn_module_stack
                if nn_module_stack:
                    rv.node.meta['nn_module_stack'] = nn_module_stack.copy()
            if 'source_fn_stack' not in rv.node.meta:
                if kind in {'call_function', 'call_method'}:
                    rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, target)]
                elif kind == 'call_module':
                    if self.parent is not None:
                        unimplemented('Invoking an nn.Module inside HigherOrderOperator')
                    rv.node.meta['source_fn_stack'] = self.source_fn_stack + [(rv.node.name, rv.node.meta['nn_module_stack'][target][1])]
        if 'stack_trace' not in rv.node.meta:
            frame_summaries: List[traceback.FrameSummary] = []
            while tx:
                frame_summaries.append(tx.frame_summary())
                tx = getattr(tx, 'parent', None)
            frame_summaries.reverse()
            msgs = traceback.StackSummary.from_list(frame_summaries).format()
            rv.node.stack_trace = ''.join(msgs)
        return rv

    def create_node(self, op, target, args=None, kwargs=None, name=None, type_expr=None):
        check_pt2_compliant_op(self.output_graph, op, target, args, kwargs)
        if self.parent is not None:
            flat_args = pytree.arg_tree_leaves(*args, **kwargs)
            for arg in flat_args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                assert arg.graph == self.graph, 'create_node using arg not from this SubgraphTracer'
        node = super().create_node(op, target, args, kwargs, name, type_expr)
        node.meta['creation_timestamp'] = self.output_graph.timestamp
        return node

    def remove_node(self, node):
        if len(node.users) > 0:
            user_graph_nodes: List[torch.fx.Node] = []
            for user in node.users.keys():
                if user.graph != self.graph:
                    user_graph_nodes.extend(reversed(list(user.graph.nodes)))
            for other_graph_node in user_graph_nodes:
                other_graph_node.graph.erase_node(other_graph_node)
        self.graph.erase_node(node)
        self.input_name_to_proxy.pop(node.name, None)

    def create_graph_input(self, name, type_expr=None, before=False, source=None):
        log.debug('create_graph_input %s %s', name, source.name() if source is not None else '(none)')
        if source is None:
            assert self.parent is not None, 'you are required to provide a source for inputs on the root tracer'
        if self.export_root:
            if not is_from_local_source(source, allow_cell_or_freevar=False):
                self.output_graph.source_to_user_stacks.setdefault(source, []).append(TracingContext.extract_stack())
        if name in self.input_name_to_proxy:
            for i in itertools.count():
                candidate_name = f'{name}_{i}'
                if candidate_name not in self.input_name_to_proxy:
                    name = candidate_name
                    break
        if self.input_name_to_proxy:
            prev_name = next(reversed(self.input_name_to_proxy))
            node = self.input_name_to_proxy[prev_name].node
            if before:
                ctx = self.graph.inserting_before(node)
            else:
                ctx = self.graph.inserting_after(node)
        else:
            ctx = self.graph.inserting_before(None)
        with ctx:
            proxy = self.create_proxy('placeholder', name, (), {}, type_expr=type_expr)
            if self.input_name_to_proxy and before:
                k, v = self.input_name_to_proxy.popitem()
                self.input_name_to_proxy[name] = proxy
                self.input_name_to_proxy[k] = v
            else:
                self.input_name_to_proxy[name] = proxy
            return proxy

    def lift_tracked_freevar_to_input(self, proxy):
        assert self.parent is not None, 'lift_tracked_freevar_to_input should not be called on root SubgraphTracer'
        if proxy in self.lifted_freevars:
            return self.lifted_freevars[proxy]
        new_proxy = self.create_graph_input(proxy.node.name)
        new_proxy.node.meta['example_value'] = proxy.node.meta['example_value']
        self.lifted_freevars[proxy] = new_proxy
        if self.parent is not None and proxy.tracer != self.parent:
            self.parent.lift_tracked_freevar_to_input(proxy)
        return new_proxy

    def maybe_lift_tracked_freevar_to_input(self, arg):
        """
        If arg is a free variable, then lift it to be an input.
        Returns the new lifted arg (if arg was a freevar), else the
        original arg.
        """
        if not isinstance(arg, torch.fx.Proxy):
            return arg
        elif arg.tracer == self:
            return arg
        return self.lift_tracked_freevar_to_input(arg)