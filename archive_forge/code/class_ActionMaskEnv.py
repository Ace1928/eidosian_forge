from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.examples.env.random_env import RandomEnv
class ActionMaskEnv(RandomEnv):
    """A randomly acting environment that publishes an action-mask each step."""

    def __init__(self, config):
        super().__init__(config)
        self._skip_env_checking = True
        assert isinstance(self.action_space, Discrete)
        self.observation_space = Dict({'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)), 'observations': self.observation_space})
        self.valid_actions = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset()
        self._fix_action_mask(obs)
        return (obs, info)

    def step(self, action):
        if not self.valid_actions[action]:
            raise ValueError(f'Invalid action ({action}) sent to env! valid_actions={self.valid_actions}')
        obs, rew, done, truncated, info = super().step(action)
        self._fix_action_mask(obs)
        return (obs, rew, done, truncated, info)

    def _fix_action_mask(self, obs):
        self.valid_actions = np.round(obs['action_mask'])
        obs['action_mask'] = self.valid_actions