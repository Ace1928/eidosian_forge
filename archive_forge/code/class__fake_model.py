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
class _fake_model:

    def __init__(self, state_space, config):
        self.state_space = state_space
        self.view_requirements = {SampleBatch.AGENT_INDEX: ViewRequirement(), SampleBatch.EPS_ID: ViewRequirement(), 'env_id': ViewRequirement(), 't': ViewRequirement(), SampleBatch.OBS: ViewRequirement(), 'state_in_0': ViewRequirement('state_out_0', shift='-50:-1', batch_repeat_value=config['model']['max_seq_len'], space=state_space), 'state_out_0': ViewRequirement(space=state_space, used_for_compute_actions=False)}

    def compile(self, *args, **kwargs):
        """Dummy method for compatibility with TorchRLModule.

                This is hit when RolloutWorker tries to compile TorchRLModule."""
        pass

    def get_initial_state(self):
        return [self.state_space.sample()]