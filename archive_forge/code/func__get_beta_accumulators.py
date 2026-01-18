from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
def _get_beta_accumulators(self):
    with ops.init_scope():
        if context.executing_eagerly():
            graph = None
        else:
            graph = ops.get_default_graph()
        return (self._get_non_slot_variable('beta1_power', graph=graph), self._get_non_slot_variable('beta2_power', graph=graph))