from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
class PackedVarAndDevice(object):
    """Holds a packed distributed variable and a device."""

    def __init__(self, var, device):
        self._var = var
        self._device = device

    def __getattr__(self, name):
        try:
            with ops.device(self._device):
                return getattr(self._var, name)
        except:
            raise

    def var(self):
        return self._var

    def value(self):
        with ops.device(self._device):
            return self._var.value()

    def read_value(self):
        with ops.device(self._device):
            return self._var.read_value()

    @property
    def initial_value(self):
        return self._var.initial_value(self._device)

    def initialized_value(self):
        with ops.device(self._device):
            return self._var.initialized_value()

    @property
    def device(self):
        return self._device

    @property
    def handle(self):
        with ops.device(self._device):
            return self._var.handle

    def on_device_handle(self):
        with ops.device(self._device):
            return self._var.get_var_on_current_device().handle

    @property
    def op(self):
        with ops.device(self._device):
            return self._var.op

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        with ops.device(self._device):
            return self._var.assign_sub(delta, use_locking, name, read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        with ops.device(self._device):
            return self._var.assign_add(delta, use_locking, name, read_value)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        with ops.device(self._device):
            return self._var.assign(value, use_locking, name, read_value)

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_sub(sparse_delta, use_locking, name)

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_add(sparse_delta, use_locking, name)

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_mul(sparse_delta, use_locking, name)

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_div(sparse_delta, use_locking, name)

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_min(sparse_delta, use_locking, name)

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_max(sparse_delta, use_locking, name)

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        with ops.device(self._device):
            return self._var.scatter_update(sparse_delta, use_locking, name)

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        with ops.device(self._device):
            return self._var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)

    def _as_graph_element(self):
        return self._var._as_graph_element()