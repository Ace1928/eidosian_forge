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
def _tf_ag_dataset_any(iterable):
    specs = nest.flatten(iterable.element_spec)
    if len(specs) != 1 or specs[0].dtype != dtypes.bool:
        raise ValueError('in graph mode, the "any" builtin only supports datasets that return bool scalars; got: {}'.format(iterable.element_spec))
    ds = iterable.filter(lambda x: x)
    ds = ds.take(1)
    ds = ds.reduce(constant_op.constant(False, dtype=dtypes.bool), lambda _, y: y)
    return ds