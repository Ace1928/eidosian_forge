import collections
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _constant_inputs(op_or_tensor):
    return all((_as_operation(i).type == u'Const' and (not _as_operation(i).control_inputs) for i in op_selector.graph_inputs(_as_operation(op_or_tensor))))