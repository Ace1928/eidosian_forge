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
def _exists_dependency(start, end):
    """Returns whether there exists a dependency chain from start to end."""
    nexts = [start]
    while nexts:
        op, nexts = (nexts[0], nexts[1:])
        for next_op in _op_dependencies(op):
            if next_op == end:
                return True
            nexts.append(next_op)
    return False