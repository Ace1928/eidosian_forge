import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication
class TPUUninitializedVariable(resource_variable_ops.UninitializedVariable):
    """UninitializedVariable component for TPU.

  Sometimes user might assign (different values) to a single component of a
  mirrored TPU variable. Thus we need to initialize_all when the assign* or read
  is invoked on a single component.
  """

    def read_value(self):
        self._lazy_scope.initialize_all()
        return super().read_value()

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        self._lazy_scope.initialize_all()
        return super().assign_sub(delta, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        self._lazy_scope.initialize_all()
        return super().assign(value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        self._lazy_scope.initialize_all()
        return super().assign_add(delta, use_locking=use_locking, name=name, read_value=read_value)