from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class AutoregressiveActionModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(AutoregressiveActionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        if action_space != Tuple([Discrete(2), Discrete(2)]):
            raise ValueError('This model only supports the [2, 2] action space')
        obs_input = tf.keras.layers.Input(shape=obs_space.shape, name='obs_input')
        a1_input = tf.keras.layers.Input(shape=(1,), name='a1_input')
        ctx_input = tf.keras.layers.Input(shape=(num_outputs,), name='ctx_input')
        context = tf.keras.layers.Dense(num_outputs, name='hidden', activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(obs_input)
        value_out = tf.keras.layers.Dense(1, name='value_out', activation=None, kernel_initializer=normc_initializer(0.01))(context)
        a1_logits = tf.keras.layers.Dense(2, name='a1_logits', activation=None, kernel_initializer=normc_initializer(0.01))(ctx_input)
        a2_context = a1_input
        a2_hidden = tf.keras.layers.Dense(16, name='a2_hidden', activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(a2_context)
        a2_logits = tf.keras.layers.Dense(2, name='a2_logits', activation=None, kernel_initializer=normc_initializer(0.01))(a2_hidden)
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.base_model.summary()
        self.action_model = tf.keras.Model([ctx_input, a1_input], [a1_logits, a2_logits])
        self.action_model.summary()

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict['obs'])
        return (context, state)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])