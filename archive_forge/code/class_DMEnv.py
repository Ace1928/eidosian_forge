import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
@PublicAPI
class DMEnv(gym.Env):
    """A `gym.Env` wrapper for the `dm_env` API."""
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, dm_env):
        super(DMEnv, self).__init__()
        self._env = dm_env
        self._prev_obs = None
        if specs is None:
            raise RuntimeError('The `specs` module from `dm_env` was not imported. Make sure `dm_env` is installed and visible in the current python environment.')

    def step(self, action):
        ts = self._env.step(action)
        reward = ts.reward
        if reward is None:
            reward = 0.0
        return (ts.observation, reward, ts.last(), False, {'discount': ts.discount})

    def reset(self, *, seed=None, options=None):
        ts = self._env.reset()
        return (ts.observation, {})

    def render(self, mode='rgb_array'):
        if self._prev_obs is None:
            raise ValueError('Environment not started. Make sure to reset before rendering.')
        if mode == 'rgb_array':
            return self._prev_obs
        else:
            raise NotImplementedError("Render mode '{}' is not supported.".format(mode))

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return _convert_spec_to_space(spec)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()
        return _convert_spec_to_space(spec)

    @property
    def reward_range(self):
        spec = self._env.reward_spec()
        if isinstance(spec, specs.BoundedArray):
            return (spec.minimum, spec.maximum)
        return (-float('inf'), float('inf'))