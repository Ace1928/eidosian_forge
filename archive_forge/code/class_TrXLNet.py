import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Any, Dict, Optional, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.layers import (
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
class TrXLNet(RecurrentNetwork):
    """A TrXL net Model described in [1]."""

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str, num_transformer_units: int, attention_dim: int, num_heads: int, head_dim: int, position_wise_mlp_dim: int):
        """Initializes a TrXLNet object.

        Args:
            num_transformer_units: The number of Transformer repeats to
                use (denoted L in [2]).
            attention_dim: The input and output dimensions of one
                Transformer unit.
            num_heads: The number of attention heads to use in parallel.
                Denoted as `H` in [3].
            head_dim: The dimension of a single(!) attention head within
                a multi-head attention unit. Denoted as `d` in [3].
            position_wise_mlp_dim: The dimension of the hidden layer
                within the position-wise MLP (after the multi-head attention
                block within one Transformer unit). This is the size of the
                first of the two layers within the PositionwiseFeedforward. The
                second layer always has size=`attention_dim`.
        """
        if log_once('trxl_net_tf'):
            deprecation_warning(old='rllib.models.tf.attention_net.TrXLNet')
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.num_transformer_units = num_transformer_units
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = model_config['max_seq_len']
        self.obs_dim = observation_space.shape[0]
        inputs = tf.keras.layers.Input(shape=(self.max_seq_len, self.obs_dim), name='inputs')
        E_out = tf.keras.layers.Dense(attention_dim)(inputs)
        for _ in range(self.num_transformer_units):
            MHA_out = SkipConnection(RelativeMultiHeadAttention(out_dim=attention_dim, num_heads=num_heads, head_dim=head_dim, input_layernorm=False, output_activation=None), fan_in_layer=None)(E_out)
            E_out = SkipConnection(PositionwiseFeedforward(attention_dim, position_wise_mlp_dim))(MHA_out)
            E_out = tf.keras.layers.LayerNormalization(axis=-1)(E_out)
        logits = tf.keras.layers.Dense(self.num_outputs, activation=tf.keras.activations.linear, name='logits')(E_out)
        self.base_model = tf.keras.models.Model([inputs], [logits])

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        observations = state[0]
        observations = tf.concat((observations, inputs), axis=1)[:, -self.max_seq_len:]
        logits = self.base_model([observations])
        T = tf.shape(inputs)[1]
        logits = logits[:, -T:]
        return (logits, [observations])

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        return [np.zeros((self.max_seq_len, self.obs_dim), np.float32)]