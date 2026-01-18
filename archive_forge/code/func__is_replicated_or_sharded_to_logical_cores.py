from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
def _is_replicated_or_sharded_to_logical_cores(self):
    """Returns whether each of the underlying variables is replicated or sharded to logical cores.

    If True, the handles of the underlying variables are not available outside a
    TPU context.
    """
    return isinstance(self._primary, tpu_replicated_variable.TPUReplicatedVariable)