import gymnasium as gym
from typing import Type
class ActionTransform(gym.ActionWrapper):

    def __init__(self, env, low, high):
        super().__init__(env)
        self._low = low
        self._high = high
        self.action_space = type(env.action_space)(self._low, self._high, env.action_space.shape, env.action_space.dtype)

    def action(self, action):
        return (action - self._low) / (self._high - self._low) * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low