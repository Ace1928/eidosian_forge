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
class AttentionWrapper(TorchModelV2, nn.Module):
    """GTrXL wrapper serving as interface for ModelV2s that set use_attention."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        if log_once('deprecate_attention_wrapper_torch'):
            deprecation_warning(old='ray.rllib.models.torch.attention_net.AttentionWrapper')
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, None, model_config, name)
        self.use_n_prev_actions = model_config['attention_use_n_prev_actions']
        self.use_n_prev_rewards = model_config['attention_use_n_prev_rewards']
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
        if self.use_n_prev_actions:
            self.num_outputs += self.use_n_prev_actions * self.action_dim
        if self.use_n_prev_rewards:
            self.num_outputs += self.use_n_prev_rewards
        cfg = model_config
        self.attention_dim = cfg['attention_dim']
        if self.num_outputs is not None:
            in_space = gym.spaces.Box(float('-inf'), float('inf'), shape=(self.num_outputs,), dtype=np.float32)
        else:
            in_space = obs_space
        self.gtrxl = GTrXLNet(in_space, action_space, None, model_config, 'gtrxl', num_transformer_units=cfg['attention_num_transformer_units'], attention_dim=self.attention_dim, num_heads=cfg['attention_num_heads'], head_dim=cfg['attention_head_dim'], memory_inference=cfg['attention_memory_inference'], memory_training=cfg['attention_memory_training'], position_wise_mlp_dim=cfg['attention_position_wise_mlp_dim'], init_gru_gate_bias=cfg['attention_init_gru_gate_bias'])
        self.num_outputs = num_outputs
        self._logits_branch = SlimFC(in_size=self.attention_dim, out_size=self.num_outputs, activation_fn=None, initializer=torch.nn.init.xavier_uniform_)
        self._value_branch = SlimFC(in_size=self.attention_dim, out_size=1, activation_fn=None, initializer=torch.nn.init.xavier_uniform_)
        self.view_requirements = self.gtrxl.view_requirements
        self.view_requirements['obs'].space = self.obs_space
        if self.use_n_prev_actions:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(SampleBatch.ACTIONS, space=self.action_space, shift='-{}:-1'.format(self.use_n_prev_actions))
        if self.use_n_prev_rewards:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(SampleBatch.REWARDS, shift='-{}:-1'.format(self.use_n_prev_rewards))

    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        assert seq_lens is not None
        wrapped_out, _ = self._wrapped_forward(input_dict, [], None)
        prev_a_r = []
        if self.use_n_prev_actions:
            prev_n_actions = input_dict[SampleBatch.PREV_ACTIONS]
            if self.model_config['_disable_action_flattening']:
                flat = flatten_inputs_to_1d_tensor(prev_n_actions, spaces_struct=self.action_space_struct, time_axis=True)
                flat = torch.reshape(flat, [flat.shape[0], -1])
                prev_a_r.append(flat)
            elif isinstance(self.action_space, Discrete):
                for i in range(self.use_n_prev_actions):
                    prev_a_r.append(one_hot(prev_n_actions[:, i].float(), space=self.action_space))
            elif isinstance(self.action_space, MultiDiscrete):
                for i in range(0, self.use_n_prev_actions, self.action_space.shape[0]):
                    prev_a_r.append(one_hot(prev_n_actions[:, i:i + self.action_space.shape[0]].float(), space=self.action_space))
            else:
                prev_a_r.append(torch.reshape(prev_n_actions.float(), [-1, self.use_n_prev_actions * self.action_dim]))
        if self.use_n_prev_rewards:
            prev_a_r.append(torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(), [-1, self.use_n_prev_rewards]))
        if prev_a_r:
            wrapped_out = torch.cat([wrapped_out] + prev_a_r, dim=1)
        input_dict['obs_flat'] = input_dict['obs'] = wrapped_out
        self._features, memory_outs = self.gtrxl(input_dict, state, seq_lens)
        model_out = self._logits_branch(self._features)
        return (model_out, memory_outs)

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        return [torch.zeros(self.gtrxl.view_requirements['state_in_{}'.format(i)].space.shape) for i in range(self.gtrxl.num_transformer_units)]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, 'Must call forward() first!'
        return torch.reshape(self._value_branch(self._features), [-1])