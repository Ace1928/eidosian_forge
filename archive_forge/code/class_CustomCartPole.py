from typing import Dict, Tuple
import argparse
import gymnasium as gym
import numpy as np
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
class CustomCartPole(gym.Env):

    def __init__(self, config):
        self.env = gym.make('CartPole-v1')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._pole_angle_vel = 0.0
        self.last_angle = 0.0

    def reset(self, *, seed=None, options=None):
        self._pole_angle_vel = 0.0
        obs, info = self.env.reset()
        self.last_angle = obs[2]
        return (obs, info)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        angle = obs[2]
        self._pole_angle_vel = 0.5 * (angle - self.last_angle) + 0.5 * self._pole_angle_vel
        info['pole_angle_vel'] = self._pole_angle_vel
        return (obs, rew, term, trunc, info)