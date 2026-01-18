from gymnasium.spaces import Box
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import (
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class DuelingQModel(TFModelV2):
    """A simple, hard-coded dueling head model."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(DuelingQModel, self).__init__(obs_space, action_space, None, model_config, name)
        self.A = tf.keras.layers.Dense(num_outputs)
        self.V = tf.keras.layers.Dense(1)

    def get_q_values(self, underlying_output):
        v = self.V(underlying_output)
        a = self.A(underlying_output)
        advantages_mean = tf.reduce_mean(a, 1)
        advantages_centered = a - tf.expand_dims(advantages_mean, 1)
        return v + advantages_centered