from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _maybe_get_unique(name):
    """Get name for a unique variable, if not `reuse=True`."""
    if variable_scope.get_variable_scope().reuse:
        return name
    vs_vars = [x.op.name for x in variable_scope.get_variable_scope().global_variables()]
    full_name = variable_scope.get_variable_scope().name + '/' + name
    if full_name not in vs_vars:
        return name
    idx = 1
    while full_name + '_%d' % idx in vs_vars:
        idx += 1
    return name + '_%d' % idx