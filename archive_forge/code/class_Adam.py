from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import gen_training_ops
class Adam(optimizer_v2.OptimizerV2):
    """Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.

  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

  Notes:

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam', **kwargs):
        super(Adam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[var_device, var_dtype]['lr_t'] * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
        apply_state[var_device, var_dtype].update(dict(lr=lr, epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(self.epsilon, var_dtype), beta_1_t=beta_1_t, beta_1_power=beta_1_power, one_minus_beta_1_t=1 - beta_1_t, beta_2_t=beta_2_t, beta_2_power=beta_2_power, one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(Adam, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = (var.device, var.dtype.base_dtype)
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        if not self.amsgrad:
            return gen_training_ops.ResourceApplyAdam(var=var.handle, m=m.handle, v=v.handle, beta1_power=coefficients['beta_1_power'], beta2_power=coefficients['beta_2_power'], lr=coefficients['lr_t'], beta1=coefficients['beta_1_t'], beta2=coefficients['beta_2_t'], epsilon=coefficients['epsilon'], grad=grad, use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'vhat')
            return gen_training_ops.ResourceApplyAdamWithAmsgrad(var=var.handle, m=m.handle, v=v.handle, vhat=vhat.handle, beta1_power=coefficients['beta_1_power'], beta2_power=coefficients['beta_2_power'], lr=coefficients['lr_t'], beta1=coefficients['beta_1_t'], beta2=coefficients['beta_2_t'], epsilon=coefficients['epsilon'], grad=grad, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = (var.device, var.dtype.base_dtype)
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'], use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = grad * grad * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'], use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
        if not self.amsgrad:
            v_sqrt = math_ops.sqrt(v_t)
            var_update = state_ops.assign_sub(var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']), use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(var, coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']), use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super(Adam, self).get_config()
        config.update({'learning_rate': self._serialize_hyperparameter('learning_rate'), 'decay': self._initial_decay, 'beta_1': self._serialize_hyperparameter('beta_1'), 'beta_2': self._serialize_hyperparameter('beta_2'), 'epsilon': self.epsilon, 'amsgrad': self.amsgrad})
        return config