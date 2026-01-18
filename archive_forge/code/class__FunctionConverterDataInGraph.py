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
class _FunctionConverterDataInGraph(_FunctionConverterData):
    """Container for ConcreteFunction-based conversion data in Graph mode."""

    def __init__(self, func, lower_control_flow, aggressive_inlining, variable_names_allowlist=None, variable_names_denylist=None, session=None):
        """Creates the conversion data for the given function.

    Args:
      func: ConcreteFunction.
      lower_control_flow: Boolean indicating whether or not to lower control
        flow ops such as If and While.
      aggressive_inlining: Boolean indicating whether or not to do aggressive
        function inlining (might be unsafe if function has stateful ops, not
        properly connected to control outputs).
      variable_names_allowlist: The set of variable names to convert (by
        default, all variables are converted).
      variable_names_denylist: The set of variable names to omit converting to
        constants.
      session: Session object.
    """
        self._session = session
        session.run(variables.global_variables_initializer())
        for op in ops.get_default_graph().get_collection(VAR_ASSIGN_COLLECTION):
            session.run(op)
        super(_FunctionConverterDataInGraph, self).__init__(func, lower_control_flow, aggressive_inlining, variable_names_allowlist, variable_names_denylist)

    def _eval(self, tensor):
        """Returns the value in the tensor. Must be implemented in sub-classes."""
        return self._session.run(tensor)