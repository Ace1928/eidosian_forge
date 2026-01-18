from typing import NamedTuple, Any
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import save_context
def _enclosing_xla_context():
    """Returns the XLAControlFlowContext, which exists inside a tpu.rewrite()."""
    graph = ops.get_default_graph()
    while graph is not None:
        context_ = graph._get_control_flow_context()
        while context_ is not None:
            if isinstance(context_, control_flow_ops.XLAControlFlowContext):
                return context_
            context_ = context_.outer_context
        graph = getattr(graph, 'outer_graph', None)
    return None