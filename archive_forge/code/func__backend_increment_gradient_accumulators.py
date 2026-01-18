import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def _backend_increment_gradient_accumulators(self, grads):

    def update_accumulator(var, grad):
        var.assign(var + grad)
    accumulators = [v.value for v in self._accumulated_gradients]

    def _distributed_tf_increment_grad_acc(distribution, grads, accumulators):
        for grad, var in zip(grads, accumulators):
            distribution.extended.update(var, update_accumulator, args=(grad,), group=False)
    tf.__internal__.distribute.interim.maybe_merge_call(_distributed_tf_increment_grad_acc, self._distribution_strategy, grads, accumulators)