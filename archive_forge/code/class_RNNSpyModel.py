import numpy as np
import pickle
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
class RNNSpyModel(RecurrentNetwork):
    capture_index = 0
    cell_size = 3

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.cell_size = RNNSpyModel.cell_size
        inputs = tf.keras.layers.Input(shape=(None,) + obs_space.shape, name='input')
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name='h')
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name='c')
        seq_lens = tf.keras.layers.Input(shape=(), name='seq_lens', dtype=tf.int32)
        lstm_out, state_out_h, state_out_c = tf.keras.layers.LSTM(self.cell_size, return_sequences=True, return_state=True, name='lstm')(inputs=inputs, mask=tf.sequence_mask(seq_lens), initial_state=[state_in_h, state_in_c])
        logits = SpyLayer(num_outputs=self.num_outputs)([inputs, lstm_out, seq_lens, state_in_h, state_in_c, state_out_h, state_out_c])
        value_out = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))(lstm_out)
        self.base_model = tf.keras.Model([inputs, seq_lens, state_in_h, state_in_c], [logits, value_out, state_out_h, state_out_c])
        self.base_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        RNNSpyModel.capture_index = 0
        model_out, value_out, h, c = self.base_model([inputs, seq_lens, state[0], state[1]])
        self._value_out = value_out
        return (model_out, [h, c])

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32), np.zeros(self.cell_size, np.float32)]