import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
class DebugCounterEnv(gym.Env):
    """Simple Env that yields a ts counter as observation (0-based).

    Actions have no effect.
    The episode length is always 15.
    Reward is always: current ts % 3.
    """

    def __init__(self, config=None):
        config = config or {}
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 100, (1,), dtype=np.float32)
        self.start_at_t = int(config.get('start_at_t', 0))
        self.i = self.start_at_t

    def reset(self, *, seed=None, options=None):
        self.i = self.start_at_t
        return (self._get_obs(), {})

    def step(self, action):
        self.i += 1
        terminated = False
        truncated = self.i >= 15 + self.start_at_t
        return (self._get_obs(), float(self.i % 3), terminated, truncated, {})

    def _get_obs(self):
        return np.array([self.i], dtype=np.float32)