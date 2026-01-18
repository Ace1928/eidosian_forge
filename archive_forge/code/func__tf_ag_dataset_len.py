from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _tf_ag_dataset_len(s):
    """Autograph override of the builtin len for dataset_ops.DataSetV2."""
    l = s.cardinality()
    msg = gen_string_ops.string_join(['len requires dataset with definitive cardinality, got ', gen_string_ops.as_string(l)])
    with ops.control_dependencies([control_flow_assert.Assert(math_ops.logical_and(math_ops.not_equal(l, dataset_ops.INFINITE), math_ops.not_equal(l, dataset_ops.UNKNOWN)), [msg])]):
        l = array_ops.identity(l)
    return l