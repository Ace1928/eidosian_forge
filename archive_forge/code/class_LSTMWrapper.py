import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import tree  # pip install dm_tree
from typing import Dict, List, Union, Tuple
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util.debug import log_once
@DeveloperAPI
class LSTMWrapper(RecurrentNetwork, nn.Module):
    """An LSTM wrapper serving as an interface for ModelV2s that set use_lstm."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        nn.Module.__init__(self)
        super(LSTMWrapper, self).__init__(obs_space, action_space, None, model_config, name)
        if self.num_outputs is None:
            self.num_outputs = int(np.product(self.obs_space.shape))
        self.cell_size = model_config['lstm_cell_size']
        self.time_major = model_config.get('_time_major', False)
        self.use_prev_action = model_config['lstm_use_prev_action']
        self.use_prev_reward = model_config['lstm_use_prev_reward']
        self.action_space_struct = get_base_struct_from_space(self.action_space)
        self.action_dim = 0
        for space in tree.flatten(self.action_space_struct):
            if isinstance(space, Discrete):
                self.action_dim += space.n
            elif isinstance(space, MultiDiscrete):
                self.action_dim += np.sum(space.nvec)
            elif space.shape is not None:
                self.action_dim += int(np.product(space.shape))
            else:
                self.action_dim += int(len(space))
        if self.use_prev_action:
            self.num_outputs += self.action_dim
        if self.use_prev_reward:
            self.num_outputs += 1
        self.lstm = nn.LSTM(self.num_outputs, self.cell_size, batch_first=not self.time_major)
        self.num_outputs = num_outputs
        self._logits_branch = SlimFC(in_size=self.cell_size, out_size=self.num_outputs, activation_fn=None, initializer=torch.nn.init.xavier_uniform_)
        self._value_branch = SlimFC(in_size=self.cell_size, out_size=1, activation_fn=None, initializer=torch.nn.init.xavier_uniform_)
        if model_config['lstm_use_prev_action']:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(SampleBatch.ACTIONS, space=self.action_space, shift=-1)
        if model_config['lstm_use_prev_reward']:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(SampleBatch.REWARDS, shift=-1)

    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        assert seq_lens is not None
        wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        prev_a_r = []
        if self.model_config['lstm_use_prev_action']:
            prev_a = input_dict[SampleBatch.PREV_ACTIONS]
            if self.model_config['_disable_action_flattening']:
                prev_a_r.append(flatten_inputs_to_1d_tensor(prev_a, spaces_struct=self.action_space_struct, time_axis=False))
            else:
                if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                    prev_a = one_hot(prev_a.float(), self.action_space)
                else:
                    prev_a = prev_a.float()
                prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))
        if self.model_config['lstm_use_prev_reward']:
            prev_a_r.append(torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(), [-1, 1]))
        if prev_a_r:
            wrapped_out = torch.cat([wrapped_out] + prev_a_r, dim=1)
        input_dict['obs_flat'] = wrapped_out
        return super().forward(input_dict, state, seq_lens)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        self._features, [h, c] = self.lstm(inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        model_out = self._logits_branch(self._features)
        return (model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)])

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        linear = next(self._logits_branch._model.children())
        h = [linear.weight.new(1, self.cell_size).zero_().squeeze(0), linear.weight.new(1, self.cell_size).zero_().squeeze(0)]
        return h

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, 'must call forward() first'
        return torch.reshape(self._value_branch(self._features), [-1])