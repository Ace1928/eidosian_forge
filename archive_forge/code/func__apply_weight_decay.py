import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def _apply_weight_decay(self, variables):
    if self.weight_decay is None:
        return

    def distributed_apply_weight_decay(distribution, variables, **kwargs):

        def weight_decay_fn(variable):
            if self._use_weight_decay(variable):
                lr = tf.cast(self.learning_rate, variable.dtype)
                wd = tf.cast(self.weight_decay, variable.dtype)
                variable.assign_sub(variable * wd * lr)
        for variable in variables:
            if isinstance(variable, backend.Variable):
                variable = variable.value
            distribution.extended.update(variable, weight_decay_fn, group=False)
    tf.__internal__.distribute.interim.maybe_merge_call(distributed_apply_weight_decay, self._distribution_strategy, variables)