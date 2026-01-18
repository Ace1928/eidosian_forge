import numpy as np
import pickle
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
class SpyLayer(tf.keras.layers.Layer):
    """A keras Layer, which intercepts its inputs and stored them as pickled."""
    output = np.array(0, dtype=np.int64)

    def __init__(self, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=num_outputs, kernel_initializer=normc_initializer(0.01))

    def call(self, inputs, **kwargs):
        """Does a forward pass through our Dense, but also intercepts inputs."""
        del kwargs
        spy_fn = tf1.py_func(self.spy, [inputs[0], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]], tf.int64, stateful=True)
        with tf1.control_dependencies([spy_fn]):
            return self.dense(inputs[1])

    @staticmethod
    def spy(inputs, seq_lens, h_in, c_in, h_out, c_out):
        """The actual spy operation: Store inputs in internal_kv."""
        if len(inputs) == 1:
            return SpyLayer.output
        ray.experimental.internal_kv._internal_kv_put('rnn_spy_in_{}'.format(RNNSpyModel.capture_index), pickle.dumps({'sequences': inputs, 'seq_lens': seq_lens, 'state_in': [h_in, c_in], 'state_out': [h_out, c_out]}), overwrite=True)
        RNNSpyModel.capture_index += 1
        return SpyLayer.output