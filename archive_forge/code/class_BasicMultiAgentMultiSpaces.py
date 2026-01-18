import argparse
import gymnasium as gym
import os
import ray
from ray import air, tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls
class BasicMultiAgentMultiSpaces(MultiAgentEnv):
    """A simple multi-agent example environment where agents have different spaces.

    agent0: obs=(10,), act=Discrete(2)
    agent1: obs=(20,), act=Discrete(3)

    The logic of the env doesn't really matter for this example. The point of this env
    is to show how one can use multi-agent envs, in which the different agents utilize
    different obs- and action spaces.
    """

    def __init__(self, config=None):
        self.agents = {'agent0', 'agent1'}
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()
        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({'agent0': gym.spaces.Box(low=-1.0, high=1.0, shape=(10,)), 'agent1': gym.spaces.Box(low=-1.0, high=1.0, shape=(20,))})
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict({'agent0': gym.spaces.Discrete(2), 'agent1': gym.spaces.Discrete(3)})
        super().__init__()

    def reset(self, *, seed=None, options=None):
        self.terminateds = set()
        self.truncateds = set()
        return ({i: self.observation_space[i].sample() for i in self.agents}, {})

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = ({}, {}, {}, {}, {})
        for i, action in action_dict.items():
            obs[i] = self.observation_space[i].sample()
            rew[i] = 0.0
            terminated[i] = False
            truncated[i] = False
            info[i] = {}
        terminated['__all__'] = len(self.terminateds) == len(self.agents)
        truncated['__all__'] = len(self.truncateds) == len(self.agents)
        return (obs, rew, terminated, truncated, info)