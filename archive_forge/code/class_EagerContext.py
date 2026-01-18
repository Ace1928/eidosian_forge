from typing import NamedTuple, Any
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import save_context
class EagerContext(NamedTuple):
    parent_graph: Any
    device_functions: Any
    colocation_stack: Any
    in_cross_replica_context: Any
    variable_policy: Any
    xla_context_id: Any