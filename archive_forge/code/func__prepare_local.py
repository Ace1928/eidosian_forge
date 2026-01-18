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
def _prepare_local(self, var_device, var_dtype, apply_state):
    super(NonFusedAdam, self)._prepare_local(var_device, var_dtype, apply_state)
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = apply_state[var_device, var_dtype]['lr_t'] * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
    apply_state[var_device, var_dtype].update(dict(lr=lr, epsilon=tensor_conversion.convert_to_tensor_v2_with_dispatch(self.epsilon, var_dtype), beta_1_t=beta_1_t, beta_1_power=beta_1_power, one_minus_beta_1_t=1 - beta_1_t, beta_2_t=beta_2_t, beta_2_power=beta_2_power, one_minus_beta_2_t=1 - beta_2_t))