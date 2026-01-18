from tensorflow.python.framework import tensor
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.training import gen_training_ops
def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
    if self._momentum:
        return super(SGD, self)._resource_apply_sparse_duplicate_indices(grad, var, indices, **kwargs)
    else:
        var_device, var_dtype = (var.device, var.dtype.base_dtype)
        coefficients = kwargs.get('apply_state', {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        return gen_resource_variable_ops.ResourceScatterAdd(resource=var.handle, indices=indices, updates=-grad * coefficients['lr_t'])