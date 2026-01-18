import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class TF2SharedWeightsModel(TFModelV2):
    """Example of weight sharing between two different TFModelV2s.

    NOTE: This will only work for tf2.x. When running with config.framework=tf,
    use SharedWeightsModel1 and SharedWeightsModel2 below, instead!

    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        global TF2_GLOBAL_SHARED_LAYER
        if TF2_GLOBAL_SHARED_LAYER is None:
            TF2_GLOBAL_SHARED_LAYER = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, name='fc1')
        inputs = tf.keras.layers.Input(observation_space.shape)
        last_layer = TF2_GLOBAL_SHARED_LAYER(inputs)
        output = tf.keras.layers.Dense(units=num_outputs, activation=None, name='fc_out')(last_layer)
        vf = tf.keras.layers.Dense(units=1, activation=None, name='value_out')(last_layer)
        self.base_model = tf.keras.models.Model(inputs, [output, vf])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        out, self._value_out = self.base_model(input_dict['obs'])
        return (out, [])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])