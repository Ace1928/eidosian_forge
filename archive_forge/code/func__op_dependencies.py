import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def _op_dependencies(op):
    """Returns the data and control dependencies of a tf.Operation combined."""
    deps = []
    for node in itertools.chain(op.inputs, op.control_inputs):
        if isinstance(node, tensor.Tensor):
            node = node.op
        assert isinstance(node, ops.Operation)
        deps.append(node)
    return deps