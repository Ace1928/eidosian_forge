import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _run_inline_graph_optimization(func, lower_control_flow, aggressive_inlining):
    """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops not
      properly connected to control outputs).

  Returns:
    GraphDef
  """
    graph_def = func.graph.as_graph_def()
    if not lower_control_flow:
        graph_def = disable_lower_using_switch_merge(graph_def)
    for function in graph_def.library.function:
        if 'api_implements' in function.attr:
            del function.attr['api_implements']
    meta_graph = export_meta_graph(graph_def=graph_def, graph=func.graph)
    for name in ['variables', 'model_variables', 'trainable_variables', 'local_variables']:
        raw_list = []
        for raw in meta_graph.collection_def['variables'].bytes_list.value:
            variable = variable_pb2.VariableDef()
            variable.ParseFromString(raw)
            variable.ClearField('initializer_name')
            raw_list.append(variable.SerializeToString())
        meta_graph.collection_def[name].bytes_list.value[:] = raw_list
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in func.inputs + func.outputs:
        fetch_collection.node_list.value.append(array.name)
    meta_graph.collection_def['train_op'].CopyFrom(fetch_collection)
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    rewrite_options.optimizers.append('function')
    if aggressive_inlining:
        rewrite_options.function_optimization = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
    return tf_optimizer.OptimizeGraph(config, meta_graph)