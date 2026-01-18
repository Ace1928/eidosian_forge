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
class TPUOnReadPolicy(values.OnReadPolicy):
    """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

    def assign_sub(self, var, *args, **kwargs):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUOnReadPolicy, self).assign_sub(var, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_sub_variable_op)(var, *args, **kwargs)

    def assign_add(self, var, *args, **kwargs):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUOnReadPolicy, self).assign_add(var, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_add_variable_op)(var, *args, **kwargs)

    def assign(self, var, *args, **kwargs):
        if tpu_util.enclosing_tpu_context() is None:
            return super(TPUOnReadPolicy, self).assign(var, *args, **kwargs)
        else:
            return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(var, *args, **kwargs)

    def scatter_sub(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_add(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_max(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_min(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_mul(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_div(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_update(self, *args, **kwargs):
        raise NotImplementedError