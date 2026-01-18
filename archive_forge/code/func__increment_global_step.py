from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _increment_global_step(increment, graph=None):
    graph = graph or ops.get_default_graph()
    global_step_tensor = get_global_step(graph)
    if global_step_tensor is None:
        raise ValueError('Global step tensor should be created by tf.train.get_or_create_global_step before calling increment.')
    global_step_read_tensor = _get_or_create_global_step_read(graph)
    with graph.as_default() as g, g.name_scope(None):
        with g.name_scope(global_step_tensor.op.name + '/'):
            with ops.control_dependencies([global_step_read_tensor]):
                return state_ops.assign_add(global_step_tensor, increment)