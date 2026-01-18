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
def _initialize_if_uninitialized(self):
    if getattr(self, '_is_lazily_initialized', False):
        return
    self._lazy_scope.initialize_all()
    self._is_lazily_initialized = True