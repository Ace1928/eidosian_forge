import argparse
import gymnasium as gym
import numpy as np
import ray
from gymnasium.spaces import Box, Discrete
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
class CustomRenderedEnv(gym.Env):
    """Example of a custom env, for which you can specify rendering behavior."""
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, config):
        self.end_pos = config.get('corridor_length', 10)
        self.max_steps = config.get('max_steps', 100)
        self.cur_pos = 0
        self.steps = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.cur_pos = 0.0
        self.steps = 0
        return ([self.cur_pos], {})

    def step(self, action):
        self.steps += 1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1.0
        elif action == 1:
            self.cur_pos += 1.0
        truncated = self.steps >= self.max_steps
        done = self.cur_pos >= self.end_pos or truncated
        return ([self.cur_pos], 10.0 if done else -0.1, done, truncated, {})

    def render(self, mode='rgb'):
        """Implements rendering logic for this env (given current state).

        You can either return an RGB image:
        np.array([height, width, 3], dtype=np.uint8) or take care of
        rendering in a window yourself here (return True then).
        For RLlib, though, only mode=rgb (returning an image) is needed,
        even when "render_env" is True in the RLlib config.

        Args:
            mode: One of "rgb", "human", or "ascii". See gym.Env for
                more information.

        Returns:
            Union[np.ndarray, bool]: An image to render or True (if rendering
                is handled entirely in here).
        """
        return np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)