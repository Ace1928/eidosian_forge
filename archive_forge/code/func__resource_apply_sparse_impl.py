import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.optimizers.legacy import optimizer_v2
from tensorflow.python.util.tf_export import keras_export
@tf.function(jit_compile=True)
def _resource_apply_sparse_impl(self, grad, var, indices, apply_state):
    var_device, var_dtype = (var.device, var.dtype.base_dtype)
    coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m.assign(m * coefficients['beta_1_t'])
    m.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
    v = self.get_slot(var, 'v')
    v_scaled_g_values = grad * grad * coefficients['one_minus_beta_2_t']
    v.assign(v * coefficients['beta_2_t'])
    v.scatter_add(tf.IndexedSlices(v_scaled_g_values, indices))
    if not self.amsgrad:
        var.assign_sub(coefficients['lr'] * m / (tf.sqrt(v) + coefficients['epsilon']))
    else:
        v_hat = self.get_slot(var, 'vhat')
        v_hat.assign(tf.maximum(v_hat, v))
        var.assign_sub(coefficients['lr'] * m / (tf.sqrt(v_hat) + coefficients['epsilon']))