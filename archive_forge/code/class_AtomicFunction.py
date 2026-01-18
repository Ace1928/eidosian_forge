import dataclasses
import traceback
import typing
from typing import Any, Dict, List, Optional, Sequence, Union
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack
class AtomicFunction:
    """A Python callable for functions in the TF Runtime.

  Provides core functionality for tf.function including:
    - automatic lifecycle management of runtime functions
    - structured inputs (including captures) and structured outputs
    - calls from both eager and graph mode
    - dependency tracking of children functions
    - runtime error interpolation to identify user code stack traces
    - compatibility with gradient infrastructure
    - control dependencies (including automatic)
  """
    __slots__ = ['_name', '_bound_context', '_function_type', '_children', '_call_options', '_cached_definition', '_cached_graph', '_generated_graph']

    def __init__(self, name: Union[str, bytes], bound_context: context.Context, function_type: function_type_lib.FunctionType, children: Optional[List['AtomicFunction']]=None, call_options: CallOptions=CallOptions(), cached_graph: Optional[func_graph_module.FuncGraph]=None):
        """Construct a new AtomicFunction.

    Args:
      name: str/bytes name of the runtime function in the bound context.
      bound_context: interface to the runtime for the AtomicFunction.
      function_type: input/output contract for the AtomicFunction
      children: list of AtomicFunctions that are needed to call this one.
      call_options: extra configuration options for the call.
      cached_graph: FuncGraph that this AtomicFunction was generated from (if
        known). Otherwise it will lazily construct a new corresponding FuncGraph
        if ever needed.
    """
        self._name = compat.as_bytes(name)
        self._bound_context = bound_context
        self._function_type = function_type
        self._children = children if children else []
        self._call_options = call_options
        self._cached_definition = None
        self._cached_graph = cached_graph
        self._generated_graph = None
        ref_key = (self._bound_context.function_scope_id, self.name)
        if ref_key not in RUNTIME_FUNCTION_REFS:
            RUNTIME_FUNCTION_REFS[ref_key] = 1
        else:
            RUNTIME_FUNCTION_REFS[ref_key] += 1

    @property
    def name(self) -> bytes:
        """Name represented in UTF-8 encoded bytes."""
        return self._name

    @property
    def function_type(self) -> function_type_lib.FunctionType:
        """Represents the input/output contract of this function."""
        return self._function_type

    @property
    def children(self) -> List['AtomicFunction']:
        """AtomicFunctions needed as dependencies for this one."""
        return self._children

    @property
    def definition(self) -> function_pb2.FunctionDef:
        """Current FunctionDef in the Runtime."""
        return self._bound_context.get_function_def(self.name)

    @property
    def attributes(self) -> Any:
        """Returns FunctionDef attributes in the Runtime."""
        attrs = self.definition.attr
        attrs.pop(attributes_lib.EAGER_RUNTIME_CONSTRUCTION_CONTEXT, None)
        return attrs

    @property
    def graph_debug_info(self) -> graph_debug_info_pb2.GraphDebugInfo:
        """A GraphDebugInfo proto mapping nodes to corresponding stack traces."""
        return self._bound_context.get_graph_debug_info(self.name)

    @property
    def call_options(self) -> CallOptions:
        """Call options declared for this AtomicFunction."""
        return self._call_options

    @property
    def graph_call_attrs(self) -> Dict[str, Any]:
        """Returns a dictionary of attributes needed to add a call in graph."""
        attrs = {'is_stateful': self.call_options.is_stateful, 'tout': [o.dtype.as_datatype_enum for o in self.function_type.flat_outputs], 'xla_compile_attr': self.cached_definition.attr.get(attributes_lib.XLA_COMPILE, None)}
        attrs.update(self._bound_context.function_call_options.as_attrs())
        return attrs

    @property
    def _c_func(self) -> Any:
        """Returns a scoped pybind object containing FunctionRecord in runtime."""
        return self._bound_context.get_c_function(self.name)

    @property
    def cached_definition(self) -> function_pb2.FunctionDef:
        """Cached FunctionDef (not guaranteed to be fresh)."""
        if self._cached_definition is None:
            self._cached_definition = self.definition
        return self._cached_definition

    @property
    def graph(self) -> func_graph_module.FuncGraph:
        """Returns a FuncGraph corresponding to the AtomicFunction."""
        if self._cached_graph:
            return self._cached_graph
        if not self._generated_graph:
            self._generated_graph = to_func_graph(self)
        return self._generated_graph

    def structured_call(self, args: Sequence[Any], kwargs: Dict[str, Any], captures: Sequence[Any]) -> Any:
        """Calls with structured tensor inputs and returns structured output."""
        bound_parameters = self.function_type.bind(*args, **kwargs)
        tensor_inputs = self.function_type.unpack_inputs(bound_parameters)
        capture_inputs = self.function_type.unpack_captures(captures)
        return self.flat_call(tensor_inputs + capture_inputs)

    def flat_call(self, args: Sequence[core.Tensor]) -> Any:
        """Calls with tensor inputs and returns the structured output."""
        flat_outputs = self(*args)
        return self.function_type.pack_output(flat_outputs)

    def __call__(self, *args: core.Tensor) -> Sequence[core.Tensor]:
        """Calls with flat tensor inputs and returns flat tensor outputs.

    Args:
      *args: arguments to call this function with.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
      FunctionAlreadyGarbageCollectedError: if the function is no longer
        available to be called because it has been garbage collected.
    """
        expected_len = len(self.cached_definition.signature.input_arg)
        if len(args) != expected_len:
            raise ValueError(f'Signature specifies {expected_len} arguments, got: {len(args)}. Expected inputs: {self.cached_definition.signature.input_arg}. Received inputs: {args}. Function Type: {self.function_type!r}')
        with InterpolateRuntimeError(self):
            with ops.control_dependencies(self._call_options.control_captures):
                with record.stop_recording():
                    if self._bound_context.executing_eagerly():
                        outputs = self._bound_context.call_function(self.name, list(args), len(self.function_type.flat_outputs))
                    else:
                        outputs = make_call_op_in_graph(self, list(args), self._bound_context.function_call_options.as_attrs())
        for i, output_type in enumerate(self.function_type.flat_outputs):
            handle_data = output_type.dtype._handle_data
            if handle_data:
                handle_data_util.set_handle_data(outputs[i], handle_data.shape_inference)
        if not self._bound_context.executing_eagerly():
            for i, output_type in enumerate(self.function_type.flat_outputs):
                outputs[i].set_shape(output_type.shape)
        return outputs

    def __del__(self):
        if self._generated_graph:
            func_graph_module.dismantle_func_graph(self._generated_graph)
        key = (self._bound_context.function_scope_id, self.name)
        RUNTIME_FUNCTION_REFS[key] -= 1
        if RUNTIME_FUNCTION_REFS[key] < 0:
            raise RuntimeError(f'AtomicFunction Refcounting for {self.name} is invalid.')
        if RUNTIME_FUNCTION_REFS[key] == 0:
            try:
                self._bound_context.remove_function(self.name)
                RUNTIME_FUNCTION_REFS.pop(key)
            except TypeError:
                pass
            except AttributeError:
                pass

    def __str__(self):
        return f'<AtomicFunction> {compat.as_str(self.name)}{self.function_type}'

    def __repr__(self):
        return f'AtomicFunction(name={self.name},\nbound_context={self._bound_context},\nfunction_type={self.function_type!r},\nchildren={self._children!s},\ncall_options={self._call_options},\ncached_graph={self._cached_graph})'