from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated
@property
def _initializer(self):
    init_ops = []
    for it in self._iterators:
        init_ops.extend(it.initialize())
    return control_flow_ops.group(init_ops)