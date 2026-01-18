import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
class FlexAgentsMultiAgent(MultiAgentEnv):
    """Env of independent agents, each of which exits after n steps."""

    def __init__(self):
        super().__init__()
        self.agents = {}
        self._agent_ids = set()
        self.agentID = 0
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    def spawn(self):
        agentID = self.agentID
        self.agents[agentID] = MockEnv(25)
        self._agent_ids.add(agentID)
        self.agentID += 1
        return agentID

    def reset(self, *, seed=None, options=None):
        self.agents = {}
        self._agent_ids = set()
        self.spawn()
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        obs = {}
        infos = {}
        for i, a in self.agents.items():
            obs[i], infos[i] = a.reset()
        return (obs, infos)

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = ({}, {}, {}, {}, {})
        for i, action in action_dict.items():
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(action)
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        if random.random() > 0.75 and len(action_dict) > 0:
            i = self.spawn()
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(action)
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        if len(self.agents) > 1 and random.random() > 0.25:
            keys = list(self.agents.keys())
            key = random.choice(keys)
            terminated[key] = True
            del self.agents[key]
        terminated['__all__'] = len(self.terminateds) == len(self.agents)
        truncated['__all__'] = len(self.truncateds) == len(self.agents)
        return (obs, rew, terminated, truncated, info)