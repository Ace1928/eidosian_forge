import contextlib
import dataclasses
import enum
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import concrete_function as concrete_function_lib
from tensorflow.python.eager.polymorphic_function import function_context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat
@dataclasses.dataclass
class TracingOptions:
    """Configuration options for tracing."""
    python_function: Callable[[Any], Any] = lambda *args, **kwargs: None
    name: str = 'function'
    polymorphic_type: Optional[function_type_lib.FunctionType] = None
    default_values: Optional[Dict[str, Any]] = None
    scope_type: ScopeType = ScopeType.NO_SCOPE
    attributes: Optional[Dict[str, Any]] = None
    autograph: bool = True
    autograph_options: Optional[Tuple[Any, ...]] = None
    reduce_retracing: bool = False
    bind_graph_to_function: bool = False
    function_cache: Optional[function_cache_lib.FunctionCache] = None
    function_captures: Optional[capture_container.FunctionCaptures] = None
    lock: Optional[threading.Lock] = None

    def __post_init__(self):
        if self.attributes:
            for attribute in self.attributes:
                if attribute not in attributes_lib.TRACING_COMPILATION_ALLOWLIST:
                    raise ValueError(f'Tracing compilation does not support `{attribute}` as an attribute.')
        if not self.polymorphic_type or self.default_values is None:
            self.polymorphic_type = function_type_lib.FunctionType.from_callable(self.python_function)
            self.default_values = function_type_lib.FunctionType.get_default_values(self.python_function)
        self._input_signature = function_type_utils.to_input_signature(self.polymorphic_type)

    @property
    def is_pure(self):
        return self.attributes and attributes_lib.IMPLEMENTS in self.attributes

    @property
    def input_signature(self):
        return self._input_signature