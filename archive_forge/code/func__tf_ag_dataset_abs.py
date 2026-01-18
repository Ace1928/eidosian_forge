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
def _tf_ag_dataset_abs(ds):
    specs = nest.flatten(ds.element_spec)
    if len(specs) == 1:
        return ds.map(math_ops.abs, num_parallel_calls=dataset_ops.AUTOTUNE)
    return ds.map(lambda *e: nest.map_structure(math_ops.abs, e), num_parallel_calls=dataset_ops.AUTOTUNE)