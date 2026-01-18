import numpy as np
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.offline import JsonReader
class CustomLossModel(TFModelV2):
    """Custom model that adds an imitation loss on top of the policy loss."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.fcnet = FullyConnectedNetwork(self.obs_space, self.action_space, num_outputs, model_config, name='fcnet')

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def value_function(self):
        return self.fcnet.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        reader = JsonReader(self.model_config['custom_model_config']['input_files'])
        input_ops = reader.tf_input_ops()
        obs = restore_original_dimensions(tf.cast(input_ops['obs'], tf.float32), self.obs_space)
        logits, _ = self.forward({'obs': obs}, [], None)
        action_dist = Categorical(logits, self.model_config)
        self.policy_loss = policy_loss
        self.imitation_loss = tf.reduce_mean(-action_dist.logp(input_ops['actions']))
        return policy_loss + 10 * self.imitation_loss

    def metrics(self):
        return {'policy_loss': self.policy_loss, 'imitation_loss': self.imitation_loss}