from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _tf_apply(self, grads, trainable_variables=None):
    """Tensorflow specific logic for apply, which handles distribution."""
    from keras.src.utils.module_utils import tensorflow as tf
    if tf.distribute.in_cross_replica_context():
        raise ValueError('apply() must be called in a replica context.')
    if tf.__internal__.distribute.strategy_supports_no_merge_call():
        self._common_apply(grads, trainable_variables=trainable_variables)
    else:

        def _handle_cross_replica(distribution, grads, trainable_variables):
            finite_per_replica = distribution.extended.call_for_each_replica(self.check_finite, args=(grads,))
            finite = distribution.experimental_local_results(finite_per_replica)[0]

            def apply_fn():
                distribution.extended.call_for_each_replica(self._stateful_handle_finite_grads, args=(grads, trainable_variables))
            ops.cond(finite, apply_fn, self._stateful_handle_non_finite_grads)
        tf.distribute.get_replica_context().merge_call(_handle_cross_replica, args=(grads, trainable_variables))