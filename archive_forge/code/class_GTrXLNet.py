import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, Optional, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules import (
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
class GTrXLNet(RecurrentNetwork, nn.Module):
    """A GTrXL net Model described in [2].

    This is still in an experimental phase.
    Can be used as a drop-in replacement for LSTMs in PPO and IMPALA.

    To use this network as a replacement for an RNN, configure your Algorithm
    as follows:

    Examples:
        >> config["model"]["custom_model"] = GTrXLNet
        >> config["model"]["max_seq_len"] = 10
        >> config["model"]["custom_model_config"] = {
        >>     num_transformer_units=1,
        >>     attention_dim=32,
        >>     num_heads=2,
        >>     memory_tau=50,
        >>     etc..
        >> }
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: Optional[int], model_config: ModelConfigDict, name: str, *, num_transformer_units: int=1, attention_dim: int=64, num_heads: int=2, memory_inference: int=50, memory_training: int=50, head_dim: int=32, position_wise_mlp_dim: int=32, init_gru_gate_bias: float=2.0):
        """Initializes a GTrXLNet.

        Args:
            num_transformer_units: The number of Transformer repeats to
                use (denoted L in [2]).
            attention_dim: The input and output dimensions of one
                Transformer unit.
            num_heads: The number of attention heads to use in parallel.
                Denoted as `H` in [3].
            memory_inference: The number of timesteps to concat (time
                axis) and feed into the next transformer unit as inference
                input. The first transformer unit will receive this number of
                past observations (plus the current one), instead.
            memory_training: The number of timesteps to concat (time
                axis) and feed into the next transformer unit as training
                input (plus the actual input sequence of len=max_seq_len).
                The first transformer unit will receive this number of
                past observations (plus the input sequence), instead.
            head_dim: The dimension of a single(!) attention head within
                a multi-head attention unit. Denoted as `d` in [3].
            position_wise_mlp_dim: The dimension of the hidden layer
                within the position-wise MLP (after the multi-head attention
                block within one Transformer unit). This is the size of the
                first of the two layers within the PositionwiseFeedforward. The
                second layer always has size=`attention_dim`.
            init_gru_gate_bias: Initial bias values for the GRU gates
                (two GRUs per Transformer unit, one after the MHA, one after
                the position-wise MLP).
        """
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.num_transformer_units = num_transformer_units
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory_inference = memory_inference
        self.memory_training = memory_training
        self.head_dim = head_dim
        self.max_seq_len = model_config['max_seq_len']
        self.obs_dim = observation_space.shape[0]
        self.linear_layer = SlimFC(in_size=self.obs_dim, out_size=self.attention_dim)
        self.layers = [self.linear_layer]
        attention_layers = []
        for i in range(self.num_transformer_units):
            MHA_layer = SkipConnection(RelativeMultiHeadAttention(in_dim=self.attention_dim, out_dim=self.attention_dim, num_heads=num_heads, head_dim=head_dim, input_layernorm=True, output_activation=nn.ReLU), fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias))
            E_layer = SkipConnection(nn.Sequential(torch.nn.LayerNorm(self.attention_dim), SlimFC(in_size=self.attention_dim, out_size=position_wise_mlp_dim, use_bias=False, activation_fn=nn.ReLU), SlimFC(in_size=position_wise_mlp_dim, out_size=self.attention_dim, use_bias=False, activation_fn=nn.ReLU)), fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias))
            attention_layers.extend([MHA_layer, E_layer])
        self.attention_layers = nn.Sequential(*attention_layers)
        self.layers.extend(attention_layers)
        self.logits = None
        self.values_out = None
        self._value_out = None
        if self.num_outputs is not None:
            self.logits = SlimFC(in_size=self.attention_dim, out_size=self.num_outputs, activation_fn=nn.ReLU)
            self.values_out = SlimFC(in_size=self.attention_dim, out_size=1, activation_fn=None)
        else:
            self.num_outputs = self.attention_dim
        for i in range(self.num_transformer_units):
            space = Box(-1.0, 1.0, shape=(self.attention_dim,))
            self.view_requirements['state_in_{}'.format(i)] = ViewRequirement('state_out_{}'.format(i), shift='-{}:-1'.format(self.memory_inference), batch_repeat_value=self.max_seq_len, space=space)
            self.view_requirements['state_out_{}'.format(i)] = ViewRequirement(space=space, used_for_training=False)

    @override(ModelV2)
    def forward(self, input_dict, state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        assert seq_lens is not None
        observations = input_dict[SampleBatch.OBS]
        B = len(seq_lens)
        T = observations.shape[0] // B
        observations = torch.reshape(observations, [-1, T] + list(observations.shape[1:]))
        all_out = observations
        memory_outs = []
        for i in range(len(self.layers)):
            if i % 2 == 1:
                all_out = self.layers[i](all_out, memory=state[i // 2])
            else:
                all_out = self.layers[i](all_out)
                memory_outs.append(all_out)
        memory_outs = memory_outs[:-1]
        if self.logits is not None:
            out = self.logits(all_out)
            self._value_out = self.values_out(all_out)
            out_dim = self.num_outputs
        else:
            out = all_out
            out_dim = self.attention_dim
        return (torch.reshape(out, [-1, out_dim]), [torch.reshape(m, [-1, self.attention_dim]) for m in memory_outs])

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[np.ndarray]:
        return []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None, 'Must call forward first AND must have value branch!'
        return torch.reshape(self._value_out, [-1])