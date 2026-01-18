import numpy as np
import tree
from gymnasium.spaces import Box
from ray.rllib.core.models.base import STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.examples.rl_module.episode_env_aware_rlm import StatefulRandomRLModule
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
class StatefulRandomPolicy(RandomPolicy):
    """A Policy that has acts randomly and has stateful view requirements."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = RLModuleConfig(action_space=self.action_space, model_config_dict={'max_seq_len': 50, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False})
        self.model = StatefulRandomRLModule(config=config)
        self.view_requirements = self.model.update_default_view_requirements(self.view_requirements)

    @override(Policy)
    def is_recurrent(self):
        return True

    def get_initial_state(self):
        if self.config.get('_enable_new_api_stack', False):
            return tree.map_structure(lambda s: convert_to_numpy(s), self.model.get_initial_state())

    @override(Policy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch['2xobs'] = sample_batch['obs'] * 2.0
        return sample_batch

    @override(Policy)
    def compute_actions_from_input_dict(self, input_dict, *args, **kwargs):
        fwd_out = self.model.forward_exploration(input_dict)
        actions = fwd_out[SampleBatch.ACTIONS]
        state_out = fwd_out[STATE_OUT]
        return (actions, state_out, {})