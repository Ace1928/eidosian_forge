import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def _distributed_tf_increment_grad_acc(distribution, grads, accumulators):
    for grad, var in zip(grads, accumulators):
        distribution.extended.update(var, update_accumulator, args=(grad,), group=False)