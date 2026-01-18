import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import (
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class BatchNormModel(TFModelV2):
    """Example of a TFModelV2 that is built w/o using tf.keras.

    NOTE: The above keras-based example model does not work with PPO (due to
    a bug in keras related to missing values for input placeholders, even
    though these input values have been provided in a forward pass through the
    actual keras Model).

    All Model logic (layers) is defined in the `forward` method (incl.
    the batch_normalization layers). Also, all variables are registered
    (only once) at the end of `forward`, so an optimizer knows which tensors
    to train on. A standard `value_function` override is used.
    """
    capture_index = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self._registered = False

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        last_layer = input_dict['obs']
        hiddens = [256, 256]
        with tf1.variable_scope('model', reuse=tf1.AUTO_REUSE):
            if isinstance(input_dict, SampleBatch):
                is_training = input_dict.is_training
            else:
                is_training = input_dict['is_training']
            for i, size in enumerate(hiddens):
                last_layer = tf1.layers.dense(last_layer, size, kernel_initializer=normc_initializer(1.0), activation=tf.nn.tanh, name='fc{}'.format(i))
                last_layer = tf1.layers.batch_normalization(last_layer, training=is_training, name='bn_{}'.format(i))
            output = tf1.layers.dense(last_layer, self.num_outputs, kernel_initializer=normc_initializer(0.01), activation=None, name='out')
            self._value_out = tf1.layers.dense(last_layer, 1, kernel_initializer=normc_initializer(1.0), activation=None, name='vf')
        if not self._registered:
            self.register_variables(self.variables())
            self.register_variables(tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='.+/model/.+'))
            self._registered = True
        return (output, [])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])