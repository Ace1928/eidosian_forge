from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _apply_gradients(self, distribution, grads_and_vars, global_step, name):
    """Unconditionally apply gradients in cross replica context."""
    update_ops = distribution.extended.call_for_each_replica(self._optimizer.apply_gradients, args=(grads_and_vars, global_step, name))
    return distribution.group(update_ops)