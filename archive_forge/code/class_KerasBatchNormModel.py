import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import (
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class KerasBatchNormModel(TFModelV2):
    """Keras version of above BatchNormModel with exactly the same structure.

    IMORTANT NOTE: This model will not work with PPO due to a bug in keras
    that surfaces when having more than one input placeholder (here: `inputs`
    and `is_training`) AND using the `make_tf_callable` helper (e.g. used by
    PPO), in which auto-placeholders are generated, then passed through the
    tf.keras. models.Model. In this last step, the connection between 1) the
    provided value in the auto-placeholder and 2) the keras `is_training`
    Input is broken and keras complains.
    Use the below `BatchNormModel` (a non-keras based TFModelV2), instead.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        inputs = tf.keras.layers.Input(shape=obs_space.shape, name='inputs')
        is_training = tf.keras.layers.Input(shape=(), dtype=tf.bool, batch_size=1, name='is_training')
        last_layer = inputs
        hiddens = [256, 256]
        for i, size in enumerate(hiddens):
            label = 'fc{}'.format(i)
            last_layer = tf.keras.layers.Dense(units=size, kernel_initializer=normc_initializer(1.0), activation=tf.nn.tanh, name=label)(last_layer)
            last_layer = tf.keras.layers.BatchNormalization()(last_layer, training=is_training[0])
        output = tf.keras.layers.Dense(units=self.num_outputs, kernel_initializer=normc_initializer(0.01), activation=None, name='fc_out')(last_layer)
        value_out = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(0.01), activation=None, name='value_out')(last_layer)
        self.base_model = tf.keras.models.Model(inputs=[inputs, is_training], outputs=[output, value_out])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, SampleBatch):
            is_training = input_dict.is_training
        else:
            is_training = input_dict['is_training']
        out, self._value_out = self.base_model([input_dict['obs'], tf.expand_dims(is_training, 0)])
        return (out, [])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])