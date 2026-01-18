import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.optimizers.legacy import optimizer_v2
from tensorflow.python.util.tf_export import keras_export
@tf.function(jit_compile=True)
def _resource_apply_dense_impl(self, grad, var, apply_state):
    var_device, var_dtype = (var.device, var.dtype.base_dtype)
    coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    alpha = coefficients['lr_t'] * tf.sqrt(1 - coefficients['beta_2_power']) / (1 - coefficients['beta_1_power'])
    m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
    v.assign_add((tf.square(grad) - v) * (1 - coefficients['beta_2_t']))
    if self.amsgrad:
        vhat = self.get_slot(var, 'vhat')
        vhat.assign(tf.maximum(vhat, v))
        v = vhat
    var.assign_sub(m * alpha / (tf.sqrt(v) + coefficients['epsilon']))