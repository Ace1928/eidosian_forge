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
class TPUDistributedVariable(TPUVariableMixin, values.DistributedVariable):
    """DistributedVariable subclass for TPUStrategy."""

    def assign_sub(self, value, use_locking=False, name=None, read_value=True):
        if values_util.is_saving_non_distributed():
            return self._primary.assign_sub(value, use_locking, name, read_value)
        return self._policy.assign_sub(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, value, use_locking=False, name=None, read_value=True):
        if values_util.is_saving_non_distributed():
            return self._primary.assign_add(value, use_locking, name, read_value)
        return self._policy.assign_add(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, value, use_locking=False, name=None, read_value=True):
        if values_util.is_saving_non_distributed():
            return self._primary.assign(value, use_locking, name, read_value)
        return self._policy.assign(self, value, use_locking=use_locking, name=name, read_value=read_value)

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_sub(sparse_delta, use_locking, name)
        return self._policy.scatter_sub(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_add(sparse_delta, use_locking, name)
        return self._policy.scatter_add(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_mul(sparse_delta, use_locking, name)
        return self._policy.scatter_mul(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_div(sparse_delta, use_locking, name)
        return self._policy.scatter_div(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_min(sparse_delta, use_locking, name)
        return self._policy.scatter_min(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_max(sparse_delta, use_locking, name)
        return self._policy.scatter_max(self, sparse_delta, use_locking=use_locking, name=name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_update(sparse_delta, use_locking, name)
        return self._policy.scatter_update(self, sparse_delta, use_locking=use_locking, name=name)