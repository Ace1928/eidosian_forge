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
class OutputGraph(Checkpointable[OutputGraphState]):
    """
    Wrapper class to hold outputs of InstructionTranslator.  Mainly the
    generated fx.Graph.

    OutputGraph is 1:1 with a frame being processed. Each frame is associated
    with some root InstructionTranslator. When user code calls a function,
    we construct a InliningInstructionTranslator that continues to write into
    the root InstructionTranslator's OutputGraph.
    """

    def __init__(self, code_options: Dict[str, Any], compiler_fn: Optional[CompilerFn], root_tx, export: bool, export_constraints, frame_state, local_scope: Scope, global_scope: Scope, f_code):
        super().__init__()
        self.tracers = [SubgraphTracer(self, export_root=export)]
        self.input_source_to_var: Dict[Source, VariableTracker] = {}
        self.export = export
        self.export_constraints = export_constraints
        self.frame_state = frame_state
        self.tensor_weakref_to_sizes_strides = WeakTensorKeyDictionary()
        self.cleanup_hooks: List[Callable[[], Any]] = []
        self.co_fields = {'co_name': f_code.co_name, 'co_filename': f_code.co_filename, 'co_firstlineno': f_code.co_firstlineno}
        self.tracked_fakes: List[TrackedFake] = []
        self.bound_symbols: Set[sympy.Symbol] = set()
        shape_env = ShapeEnv(tracked_fakes=self.tracked_fakes, allow_scalar_outputs=config.capture_scalar_outputs, allow_dynamic_output_shape_ops=config.capture_dynamic_output_shape_ops, co_fields=self.co_fields)
        fake_mode = torch._subclasses.FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True if self.export else False)
        self.tracing_context: TracingContext = TracingContext(fake_mode)
        self.init_ambient_guards()
        self.tracked_fakes_id_to_source: Dict[int, List[Source]] = collections.defaultdict(list)
        self.param_name_to_source: Optional[Dict[str, Source]] = dict()
        self.side_effects = SideEffects()
        self.code_options = dict(code_options)
        self.output_instructions: List[Instruction] = []
        self.timestamp = 0
        self.register_finalizer_fns: List[Callable[[fx.GraphModule], None]] = []
        self.compiler_fn: Optional[CompilerFn] = compiler_fn
        self.global_scope = global_scope
        self.local_scope = local_scope
        self.root_tx = root_tx
        from torch._dynamo.symbolic_convert import InstructionTranslatorBase
        self.source_to_user_stacks: Dict[Source, List[traceback.StackSummary]] = {}
        self._current_tx: List[InstructionTranslatorBase] = []
        self.cleanups: List[CleanupHook] = []
        self.should_exit = False
        self.random_values_var = None
        self.unspec_variable_map: Dict[str, UnspecializedPythonVariable] = {}
        self.torch_function_enabled = torch._C._is_torch_function_enabled()
        self.has_user_defined_allowed_in_graph = False
        self.non_compliant_ops: Set[torch._ops.OpOverload] = set({})
        self.compliant_custom_ops: Set[torch._ops.OpOverload] = set({})
        self.save_global_state()

    def init_ambient_guards(self):
        self.guards.add(ShapeEnvSource().make_guard(GuardBuilder.SHAPE_ENV))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DETERMINISTIC_ALGORITHMS))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.GRAD_MODE))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.DEFAULT_DEVICE))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.TORCH_FUNCTION_STATE))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.BACKEND_MATCH))
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.CONFIG_HASH_MATCH))

    def guard_has_graph_break(self):
        self.guards.add(GlobalStateSource().make_guard(GuardBuilder.HAS_GRAPH_BREAK))

    def add_cleanup_hook(self, fn: Callable[[], Any]):
        self.cleanup_hooks.append(fn)

    def call_cleanup_hooks(self):
        for hook in reversed(self.cleanup_hooks):
            hook()
        self.cleanup_hooks.clear()

    @property
    def root_tracer(self):
        return self.tracers[0]

    @property
    def current_tracer(self):
        return self.tracers[-1]

    def is_root_tracer(self):
        return len(self.tracers) == 1

    @property
    def graph(self):
        return self.current_tracer.graph

    @graph.setter
    def graph(self, value):
        self.current_tracer.graph = value

    @property
    def input_name_to_proxy(self):
        return self.current_tracer.input_name_to_proxy

    @property
    def real_value_cache(self):
        return self.current_tracer.real_value_cache

    def create_proxy(self, *args, **kwargs):
        return self.current_tracer.create_proxy(*args, **kwargs)

    def create_node(self, *args, **kwargs):
        return self.current_tracer.create_node(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        return self.current_tracer.remove_node(*args, **kwargs)

    @contextlib.contextmanager
    def subtracer(self, source_target, prior_tracer):
        new_scope_ctx = enter_new_scope()
        try:
            if prior_tracer:
                assert prior_tracer.parent is self.current_tracer
            new_scope_ctx.__enter__()
            tracer = prior_tracer if prior_tracer else SubgraphTracer(self, parent=self.current_tracer, source_target=source_target)
            self.tracers.append(tracer)
            yield tracer
        finally:
            new_scope_ctx.__exit__(None, None, None)
            self.tracers.pop()

    @property
    def output(self):
        return self

    @property
    def fake_mode(self):
        return self.tracing_context.fake_mode

    @property
    def shape_env(self):
        return self.tracing_context.fake_mode.shape_env

    @property
    def guards(self) -> torch._guards.GuardsSet:
        return self.tracing_context.guards_context.dynamo_guards

    @property
    def nn_modules(self) -> Dict[str, Any]:
        return self.tracing_context.module_context.nn_modules

    def save_global_state(self, out=None):
        """
        Saves to out if it is provided. Else saves to the tracing context's global_state.
        """
        global_state = out if out is not None else self.tracing_context.global_context.global_state
        global_state['torch_function_enabled'] = (self.set_torch_function_state, self.torch_function_enabled)
        global_state['grad_enabled'] = (torch.set_grad_enabled, torch.is_grad_enabled())
        global_state['autocast_enabled'] = (torch.set_autocast_enabled, torch.is_autocast_enabled())
        global_state['autocast_cpu_enabled'] = (torch.set_autocast_cpu_enabled, torch.is_autocast_cpu_enabled())
        global_state['autocast_gpu_dtype'] = (torch.set_autocast_gpu_dtype, torch.get_autocast_gpu_dtype())
        global_state['autocast_cpu_dtype'] = (torch.set_autocast_cpu_dtype, torch.get_autocast_cpu_dtype())
        global_state['autocast_cache_enabled'] = (torch.set_autocast_cache_enabled, torch.is_autocast_cache_enabled())

    def push_tx(self, tx):
        self._current_tx.append(tx)

    def pop_tx(self):
        return self._current_tx.pop()

    @property
    def current_tx(self):
        return self.root_tx if not self._current_tx else self._current_tx[-1]

    def copy_graphstate(self) -> OutputGraphState:
        """Create a checkpoint of the current state by copying everything"""
        assert self.param_name_to_source is not None
        guards_graph_state = self.tracing_context.guards_context.copy_graphstate()
        module_state = self.tracing_context.module_context.copy_graphstate()
        global_state = self.tracing_context.global_context.copy_graphstate()
        state = OutputGraphState(dict(self.input_source_to_var), list(self.tracked_fakes), guards_graph_state, module_state, list(self.register_finalizer_fns), global_state, dict(self.param_name_to_source), self.side_effects.clone(), self.timestamp, set(self.non_compliant_ops), set(self.compliant_custom_ops))
        self.timestamp += 1
        return state

    def restore_graphstate(self, state: OutputGraphState):
        """Restore a checkpoint created by self.copy_graphstate()"""
        self.input_source_to_var, self.tracked_fakes, guards_state, module_state, self.register_finalizer_fns, global_state, self.param_name_to_source, self.side_effects, self.timestamp, self.non_compliant_ops, self.compliant_custom_ops = state
        self.tracing_context.guards_context.restore_graphstate(guards_state)
        self.tracing_context.module_context.restore_graphstate(module_state)
        self.tracing_context.global_context.restore_graphstate(global_state)
        removed_nodes = 0
        for node in reversed(list(self.graph.nodes)):
            if node.meta['creation_timestamp'] > self.timestamp and node.op != 'placeholder':
                if 'example_value' in node.meta:
                    del node.meta['example_value']
                self.remove_node(node)
                self.real_value_cache.pop(node, None)
                removed_nodes += 1
        log.debug('restore_graphstate: removed %s nodes', removed_nodes)

    def add_symbol_bindings(self, arg: GraphArg):
        if self.export:
            return
        assert arg.fake_tensor is not None

        def bind_symint(s, prop):
            if not (is_symbolic(s) and isinstance(s.node.expr, sympy.Symbol)):
                return
            s0 = s.node.expr
            if s0 in self.bound_symbols:
                return
            self.bound_symbols.add(s0)
            log.debug('bind_symint %s %s', s, prop.name())
            proxy = self.root_tracer.create_graph_input(str(s0), torch.SymInt, before=True, source=prop)
            proxy.node.meta['example_value'] = s
            proxy.node.meta['grapharg'] = GraphArg(prop, s, is_unspecialized=False, fake_tensor=None, is_tensor=False)

        def handle_tensor(t, src):
            for i, s in enumerate(t.size()):
                bind_symint(s, TensorPropertySource(src, TensorProperty.SIZE, i))
            for i, s in enumerate(t.stride()):
                bind_symint(s, TensorPropertySource(src, TensorProperty.STRIDE, i))
            bind_symint(t.storage_offset(), TensorPropertySource(src, TensorProperty.STORAGE_OFFSET))
            if is_traceable_wrapper_subclass(t):
                attrs, ctx = t.__tensor_flatten__()
                for attr in attrs:
                    inner_t = getattr(t, attr)
                    handle_tensor(inner_t, AttrSource(src, attr))
        handle_tensor(arg.fake_tensor, arg.source)

    def count_calls(self):
        return count_calls(self.graph)

    def is_empty_graph(self):
        return len(list(self.graph.nodes)) == 0

    def get_submodule(self, keys):
        assert keys
        obj: Union[torch.nn.Module, Dict[str, torch.nn.Module]] = self.nn_modules
        for k in keys.split('.'):
            if isinstance(obj, dict):
                obj = obj[k]
            else:
                obj = getattr(obj, k)
        return obj

    def new_var(self, name='tmp'):
        existing = set(self.code_options['co_varnames'])
        for i in itertools.count():
            var = f'{name}_{i}'
            if var not in existing:
                self.code_options['co_varnames'] += (var,)
                return var

    def update_co_names(self, name):
        """Ensure self.code_options.co_names contains name"""
        if name not in self.code_options['co_names']:
            self.code_options['co_names'] += (name,)

    @staticmethod
    def module_key_name(*names):
        name = '_'.join(map(str, names))
        name = re.sub("^[GL]\\['?(.*?)'?\\]$", '\\1', name)
        name = re.sub('\\[(\\d+)\\]', '_\\g<1>', name)
        name = re.sub('[^a-zA-Z0-9]', '_', name)
        if not name or not name[0].isalpha():
            name = 'sub' + name
        return name

    def register_attr_or_module(self, target: Union[torch.nn.Module, torch.Tensor, Any], *names, **options):
        if is_dynamic_nn_module(target):
            return variables.UnspecializedNNModuleVariable(target, **options)
        options = dict(options)
        assert 'source' in options
        source = options['source']
        assert not isinstance(source, ParamBufferSource)
        if isinstance(target, torch.Tensor):
            tracer = self.current_tracer
            if not self.is_root_tracer():
                tracer = self.root_tracer
            if not is_constant_source(source):
                install_guard(source.make_guard(GuardBuilder.TENSOR_MATCH))
            if get_static_address_type(target) == 'guarded':
                install_guard(source.make_guard(GuardBuilder.DATA_PTR_MATCH))

            def wrap_name(module_key):
                assert self.param_name_to_source is not None
                self.param_name_to_source[module_key] = source
                return wrap_fx_proxy(self.root_tx, tracer.create_proxy('get_attr', module_key, tuple(), {}), example_value=target, **options)
        elif isinstance(target, torch.nn.Module):
            assert isinstance(target, torch.nn.Module)
            install_guard(source.make_guard(GuardBuilder.NN_MODULE))

            def wrap_name(module_key):
                return NNModuleVariable(type(target), module_key, **options)
        elif isinstance(target, (torch.SymInt, torch.SymFloat)):

            def wrap_name(module_key):
                return SymNodeVariable.create(self, self.create_proxy('get_attr', module_key, tuple(), {}), sym_num=target, **options)
        else:

            def wrap_name(module_key):
                self.output.update_co_names(module_key)
                self.global_scope[module_key] = target
                return VariableBuilder(self, ConstantSource(source_name=module_key))(target)
        for k, v in self.nn_modules.items():
            if v is target:
                return wrap_name(k)
        name = OutputGraph.module_key_name(*names)
        base = name
        for i in itertools.count():
            if name not in self.nn_modules:
                self.nn_modules[name] = target
                if isinstance(target, torch.nn.Module):

                    def register_leaf_name(leaf_name):
                        assert self.param_name_to_source is not None
                        new_source = ParamBufferSource(source, leaf_name)
                        new_name = f'{name}.{leaf_name}'
                        self.param_name_to_source[new_name] = new_source
                    if hasattr(target, '_parameters'):
                        for leaf_name, _ in target.named_parameters():
                            register_leaf_name(leaf_name)
                    if hasattr(target, '_buffers'):
                        for leaf_name, _ in target.named_buffers():
                            register_leaf_name(leaf_name)
                return wrap_name(name)
            name = f'{base}_{i}'
        raise AssertionError('unreachable')

    def compile_subgraph(self, tx, partial_convert=False, reason: Optional[GraphCompileReason]=None, compile_return_value=False):
        """
        Generate a subgraph to continue execution on user code.
        Automatically restore live variables.
        """
        assert reason is not None
        from .decorators import disable
        self.partial_convert = partial_convert
        self.compile_subgraph_reason = reason
        self.should_exit = True
        if not compile_return_value:
            self.guard_has_graph_break()
        log.debug('COMPILING GRAPH due to %s', reason)
        if not all((block.can_restore() for block in tx.block_stack)):
            unimplemented('compile_subgraph with block_depth != 0')
        prefix_insts: List[Instruction] = []
        if sys.version_info >= (3, 11):
            for inst in tx.prefix_insts:
                if inst.opname == 'MAKE_CELL':
                    prefix_insts.append(create_instruction('MAKE_CELL', argval=inst.argval))
                elif inst.opname == 'COPY_FREE_VARS':
                    prefix_insts.append(create_instruction('COPY_FREE_VARS', arg=len(tx.code_options['co_freevars'])))
                else:
                    prefix_insts.append(copy.copy(inst))

        def append_prefix_insts():
            self.add_output_instructions(prefix_insts)
            prefix_insts.clear()
        for block in reversed(tx.block_stack):
            block.exit(tx)
        self.cleanup_graph()
        tx.prune_dead_locals()
        stack_values = list(tx.stack)
        root = FakeRootModule(self.nn_modules)
        restore_vars = []
        val_to_names: Dict[VariableTracker, List[str]] = {}
        if stack_values:
            val_to_names[stack_values[-1]] = list()
        for k, v in tx.symbolic_locals.items():
            if isinstance(v.source, LocalSource) and v.source.local_name == k:
                continue
            if v not in val_to_names:
                val_to_names[v] = list()
            val_to_names[v].append(k)
        for v in val_to_names.keys():
            restore_vars.extend(val_to_names[v])
            stack_values.extend([v] * len(val_to_names[v]))
        if len(tx.random_calls) > 0:
            append_prefix_insts()
            random_calls_instructions = []
            self.random_values_var = self.new_var('random_values')
            rand_fn_name = unique_id('__gen_rand_values')
            rand_fn = disable(_get_gen_rand_values_fn(tx.random_calls))
            self.install_global(rand_fn_name, rand_fn)
            codegen = PyCodegen(tx, root)
            random_calls_instructions.extend(codegen.load_function_name(rand_fn_name, True))
            random_calls_instructions.extend(create_call_function(0, False))
            random_calls_instructions.append(codegen.create_store(tx.output.random_values_var))
            self.add_output_instructions(random_calls_instructions)
        if stack_values and all((not isinstance(v, (UnspecializedPythonVariable, NumpyNdarrayVariable, TensorWithTFOverrideVariable)) for v in stack_values)) and all((isinstance(x, TensorVariable) for x in stack_values)) and (len(set(stack_values)) == len(stack_values)) and self.side_effects.is_empty():
            append_prefix_insts()
            self.add_output_instructions(self.compile_and_call_fx_graph(tx, list(reversed(stack_values)), root) + [create_instruction('UNPACK_SEQUENCE', arg=len(stack_values))])
        else:
            graph_output_var = self.new_var('graph_out')
            pass1 = PyCodegen(tx, root, graph_output_var)
            self.side_effects.codegen_hooks(pass1)
            self.side_effects.codegen_save_tempvars(pass1)
            pass1.restore_stack(stack_values)
            self.side_effects.codegen_update_mutated(pass1)
            pass2 = PyCodegen(tx, root, graph_output_var, tempvars={val: None for val, count in pass1.uses.items() if count > 1})
            self.side_effects.codegen_hooks(pass2)
            self.side_effects.codegen_save_tempvars(pass2)
            pass2.restore_stack(stack_values)
            self.side_effects.codegen_update_mutated(pass2)
            output = []
            if count_calls(self.graph) != 0 or len(pass2.graph_outputs) != 0:
                output.extend(self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root))
                if len(pass2.graph_outputs) != 0:
                    output.append(pass2.create_store(graph_output_var))
                else:
                    output.append(create_instruction('POP_TOP'))
            append_prefix_insts()
            self.add_output_instructions(output + pass2.get_instructions())
        self.add_output_instructions([PyCodegen(tx).create_store(var) for var in reversed(restore_vars)])

    def cleanup_graph(self):
        """
        Remove "creation_timestamp" from node meta

        Remove this pattern from the graph:
            torch._C._set_grad_enabled(False)
            torch._C._set_grad_enabled(True)
        """
        assert self.should_exit
        nodes = list(self.graph.nodes)
        for node in nodes:
            node.meta.pop('creation_timestamp', None)
        grad_enabled = torch.is_grad_enabled()
        for node1, node2 in zip(nodes, nodes[1:]):
            if node1.target is torch._C._set_grad_enabled and tuple(node1.args) == (not grad_enabled,) and (not node1._erased):
                grad_enabled = node1.args[0]
                if node2.target is torch._C._set_grad_enabled and tuple(node2.args) == (not grad_enabled,) and (not node2._erased):
                    grad_enabled = node2.args[0]
                    self.graph.erase_node(node1)
                    self.graph.erase_node(node2)

    def get_graph_sizes_log_str(self, name):
        graph_sizes_str = 'TRACED GRAPH TENSOR SIZES\n'
        graph_sizes_str += f'===== {name} =====\n'
        for node in self.graph.nodes:
            example_value = node.meta.get('example_value', None)
            if isinstance(example_value, torch._subclasses.FakeTensor):
                size = example_value.size()
                graph_sizes_str += f'{node.name}: {tuple(size)}\n'
                concrete_size = []
                has_symint = False
                for sz in size:
                    if isinstance(sz, int):
                        concrete_size.append(sz)
                    elif isinstance(sz, torch.SymInt):
                        has_symint = True
                        concrete_size.append(sz.node.hint)
                    else:
                        break
                else:
                    if has_symint:
                        graph_sizes_str += f'{node.name} (concrete): {tuple(concrete_size)}\n'
        return graph_sizes_str

    @contextlib.contextmanager
    def restore_global_state(self):
        """
        Momentarily restores the global state to what it was prior to tracing the current output
        """
        prior_global_state = self.tracing_context.global_context.copy_graphstate()
        current_global_state: Dict[str, Tuple[Any, bool]] = {}
        self.save_global_state(out=current_global_state)
        try:
            self.tracing_context.global_context.restore_graphstate(prior_global_state)
            yield
        finally:
            self.tracing_context.global_context.restore_graphstate(GlobalContextCheckpointState(current_global_state))

    @torch._guards.TracingContext.clear_frame()
    def compile_and_call_fx_graph(self, tx, rv, root):
        """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
        from .decorators import disable
        assert self.should_exit
        name = unique_id('__compiled_fn')
        assert isinstance(rv, list)
        assert isinstance(root, FakeRootModule)
        self.create_node('output', 'output', (self.current_tracer.create_arg(tuple((x.as_proxy() for x in rv))),), {})
        self.insert_deferred_runtime_asserts(root, name)
        self.remove_unused_graphargs()
        ncalls = count_calls(self.graph)
        counters['stats']['calls_captured'] += ncalls
        self.real_value_cache.clear()
        gm = fx.GraphModule(root, self.graph)
        for register_finalizer in self.register_finalizer_fns:
            register_finalizer(gm)
        gm.compile_subgraph_reason = self.compile_subgraph_reason
        graph_code_log.debug('%s', lazy_format_graph_code(name, gm))
        graph_tabular_log.debug('%s', lazy_format_graph_tabular(name, gm))
        graph_sizes_log.debug('%s', LazyString(lambda: self.get_graph_sizes_log_str(name)))
        self.call_cleanup_hooks()
        old_fake_mode = self.tracing_context.fake_mode
        if not self.export:
            backend_fake_mode = torch._subclasses.FakeTensorMode(shape_env=old_fake_mode.shape_env)
            self.tracing_context.fake_mode = backend_fake_mode
        with self.restore_global_state():
            compiled_fn = self.call_user_compiler(gm)
        compiled_fn = disable(compiled_fn)
        counters['stats']['unique_graphs'] += 1
        self.install_global(name, compiled_fn)
        cg = PyCodegen(tx)
        cg.make_call_generated_code(name)
        return cg.get_instructions()

    @property
    def placeholders(self) -> List[fx.Node]:
        r = []
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                r.append(node)
                continue
            break
        return r

    @property
    def graphargs(self) -> List[GraphArg]:
        return [node.meta['grapharg'] for node in self.placeholders]

    @dynamo_timed(phase_name='backend_compile')
    def call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:
        assert self.compiler_fn is not None
        tot = 0
        placeholders = []
        for node in gm.graph.nodes:
            if node.op in ('call_function', 'call_method', 'call_module'):
                tot += 1
            if node.op == 'placeholder':
                placeholders.append(node)
        increment_op_count(tot)
        for pl in placeholders:
            arg = pl.meta['grapharg']
            pl._dynamo_source = arg.source
        gm._param_name_to_source = self.param_name_to_source
        gm._source_to_user_stacks = self.source_to_user_stacks
        try:
            name = self.compiler_fn.__name__ if hasattr(self.compiler_fn, '__name__') else ''
            _step_logger()(logging.INFO, f'calling compiler function {name}')
            compiler_fn = self.compiler_fn
            if config.verify_correctness:
                compiler_fn = WrapperBackend(compiler_fn)
            compiled_fn = compiler_fn(gm, self.example_inputs())
            _step_logger()(logging.INFO, f'done compiler function {name}')
            assert callable(compiled_fn), 'compiler_fn did not return callable'
        except exceptions_allowed_to_be_fallback as e:
            if self.has_user_defined_allowed_in_graph:
                raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(e.__traceback__) from None
            msg = f'Backend compiler failed with a fake tensor exception at \n{self.root_tx.format_frame_summary()}Adding a graph break.'
            unimplemented_with_warning(e, self.root_tx.f_code, msg)
        except SkipFrame as e:
            raise e
        except Exception as e:
            raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(e.__traceback__) from None
        signpost_event('dynamo', 'OutputGraph.call_user_compiler', {**self.co_fields, 'op_count': tot, 'node_count': len(gm.graph.nodes), 'input_count': len(placeholders)})
        return compiled_fn

    def example_inputs(self) -> List[torch.Tensor]:
        result = []
        for arg in self.graphargs:
            result.append(arg.example)
        return result

    def remove_unused_graphargs(self) -> None:
        assert self.should_exit
        for node in reversed(list(self.graph.nodes)):
            if len(list(node.users)) == 0:
                if node.op == 'get_attr':
                    self.remove_node(node)
                elif node.op == 'call_function' and node.target is operator.getitem:
                    self.remove_node(node)

        def placeholder_binds_symbol(node):
            arg = node.meta['grapharg']
            example = arg.example
            if isinstance(example, torch.SymInt) and isinstance(example.node.expr, sympy.Symbol):
                return example.node.expr
            return None

        def remove_unused(node):
            log.debug('REMOVE UNUSED GRAPHARG %s', node.meta['grapharg'].source.name())
            del node.meta['grapharg']
            self.remove_node(node)
            self.real_value_cache.pop(node, None)
        used_symbols = set()
        recheck_placeholders = []
        for node in self.placeholders:
            binds_symbol = placeholder_binds_symbol(node) is not None
            if binds_symbol:
                if not node.users:
                    recheck_placeholders.append(node)
            elif not node.users:
                remove_unused(node)
            else:
                arg = node.meta['grapharg']
                fake = arg.fake_tensor if arg.fake_tensor is not None else arg.example
                used_symbols |= free_symbols(fake)
        for node in recheck_placeholders:
            symbol = placeholder_binds_symbol(node)
            if symbol is not None:
                if symbol not in used_symbols:
                    remove_unused(node)
                else:
                    used_symbols.remove(symbol)

    def insert_deferred_runtime_asserts(self, root, name) -> None:
        """
        During tracing, we may have discovered that some data-dependent values
        had runtime assert on them; e.g., torch.empty(x.item()) induces a runtime
        that x.item() >= 0.  This asserts can happen unpredictably during fake
        tensor propagation, so we cannot conveniently insert them into the FX graph
        when they occur.  Instead, we accumulate them in the ShapeEnv, and in this
        pass insert them into the graph as proper tests.
        """
        ras_by_symbol = self.shape_env.deferred_runtime_asserts.copy()
        if not any((ras for ras in ras_by_symbol.values())):
            return
        gm = fx.GraphModule(root, self.graph)
        graph_code_log.debug('%s', lazy_format_graph_code(f'pre insert_deferred_runtime_asserts {name}', gm))
        symbol_to_proxy = {}
        placeholders = set()
        last_placeholder = None
        for node in self.graph.nodes:
            if node.op != 'placeholder':
                last_placeholder = node
                break
            placeholders.add(node)
        assert last_placeholder is not None
        needed_symbols: Set[sympy.Symbol] = set()
        for ras in ras_by_symbol.values():
            for ra in ras:
                needed_symbols.update(free_symbols(ra.expr))
        log.debug('needed_symbols = %s', needed_symbols)
        for node in self.graph.nodes:
            with self.graph.inserting_before(node.next if node not in placeholders else last_placeholder.next):
                if 'example_value' not in node.meta:
                    continue
                defs = []

                def match_symbol(symint, cb):
                    if isinstance(symint, torch.SymInt) and isinstance(symint.node, SymNode) and isinstance((s := symint.node.expr), sympy.Symbol) and (s not in symbol_to_proxy) and (s in needed_symbols):
                        symbol_to_proxy[s] = fx.Proxy(cb())
                        log.debug('symbol_to_proxy[%s] = %s', s, symbol_to_proxy[s])
                        defs.append(s)
                match_symbol(node.meta['example_value'], lambda: node)
                if isinstance((t := node.meta['example_value']), torch.Tensor):
                    for i, s in enumerate(t.size()):
                        match_symbol(s, lambda: self.graph.call_method('size', (node, i)))
                    for i, s in enumerate(t.stride()):
                        match_symbol(s, lambda: self.graph.call_method('stride', (node, i)))
                    match_symbol(t.storage_offset(), lambda: self.graph.call_method('storage_offset', (node,)))
                for i0 in defs:
                    ras = ras_by_symbol.pop(i0, [])
                    for ra in ras:
                        log.debug('inserting runtime assert %s', ra.expr)
                        fvs = free_symbols(ra.expr)
                        missing = fvs - symbol_to_proxy.keys()
                        if missing:
                            i1 = sorted(missing)[0]
                            assert self.shape_env.is_unbacked_symint(i1), i1
                            ras_by_symbol.setdefault(i1, []).append(ra)
                        else:
                            res = sympy_interp(PythonReferenceAnalysis, symbol_to_proxy, ra.expr).node
                            res2 = self.graph.call_function(torch.ops.aten.scalar_tensor.default, (res,))
                            self.graph.call_function(torch.ops.aten._assert_async.msg, (res2, f'Deferred runtime assertion failed {ra.expr}'))

    def add_output_instructions(self, prefix: List[Instruction]) -> None:
        """
        We call this on the creation of a new compiled subgraph that is inserted
        before user code.
        """
        self.output_instructions.extend(prefix)
        self.should_exit = True

    def install_global(self, name, value) -> None:
        self.cleanups.append(CleanupHook.create(self.global_scope, name, value))

    def cleanup(self) -> None:
        self.root_tx = None
        self.nn_modules.clear()
        self.param_name_to_source = None
        for node in self.graph.nodes:
            if 'grapharg' in node.meta:
                del node.meta['grapharg']
        self.real_value_cache.clear()
        self.input_name_to_proxy.clear()
        self.side_effects.clear()
        self.register_finalizer_fns.clear()

    def set_torch_function_state(self, enabled: bool) -> None:
        self.torch_function_enabled = enabled

    def add_graph_finalizer(self, register_finalizer: Callable[[fx.GraphModule], None]) -> None:
        self.register_finalizer_fns.append(register_finalizer)