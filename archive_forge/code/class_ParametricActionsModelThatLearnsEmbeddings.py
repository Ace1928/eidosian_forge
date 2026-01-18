from gymnasium.spaces import Box
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
class ParametricActionsModelThatLearnsEmbeddings(DistributionalQTFModel):
    """Same as the above ParametricActionsModel.

    However, this version also learns the action embeddings.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(4,), action_embed_size=2, **kw):
        super(ParametricActionsModelThatLearnsEmbeddings, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        action_ids_shifted = tf.constant(list(range(1, num_outputs + 1)), dtype=tf.float32)
        obs_cart = tf.keras.layers.Input(shape=true_obs_shape, name='obs_cart')
        valid_avail_actions_mask = tf.keras.layers.Input(shape=(num_outputs,), name='valid_avail_actions_mask')
        self.pred_action_embed_model = FullyConnectedNetwork(Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size, model_config, name + '_pred_action_embed')
        pred_action_embed, _ = self.pred_action_embed_model({'obs': obs_cart})
        _value_out = self.pred_action_embed_model.value_function()
        intent_vector = tf.expand_dims(pred_action_embed, 1)
        valid_avail_actions = action_ids_shifted * valid_avail_actions_mask
        valid_avail_actions_embed = tf.keras.layers.Embedding(input_dim=num_outputs + 1, output_dim=action_embed_size, name='action_embed_matrix')(valid_avail_actions)
        action_logits = tf.reduce_sum(valid_avail_actions_embed * intent_vector, axis=2)
        inf_mask = tf.maximum(tf.math.log(valid_avail_actions_mask), tf.float32.min)
        action_logits = action_logits + inf_mask
        self.param_actions_model = tf.keras.Model(inputs=[obs_cart, valid_avail_actions_mask], outputs=[action_logits, _value_out])
        self.param_actions_model.summary()

    def forward(self, input_dict, state, seq_lens):
        valid_avail_actions_mask = input_dict['obs']['valid_avail_actions_mask']
        action_logits, self._value_out = self.param_actions_model([input_dict['obs']['cart'], valid_avail_actions_mask])
        return (action_logits, state)

    def value_function(self):
        return self._value_out