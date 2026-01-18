import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
class SometimesZeroAgentsMultiAgent(MultiAgentEnv):
    """Multi-agent env in which sometimes, no agent acts.

    At each timestep, we determine, which agents emit observations (and thereby request
    actions). This set of observing (and action-requesting) agents could be anything
    from the empty set to the full set of all agents.

    For simplicity, all agents terminate after n timesteps.
    """

    def __init__(self, num=3):
        super().__init__()
        self.num_agents = num
        self.agents = [MockEnv(25) for _ in range(self.num_agents)]
        self._agent_ids = set(range(self.num_agents))
        self._observations = {}
        self._infos = {}
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.terminateds = set()
        self.truncateds = set()
        self._observations = {}
        self._infos = {}
        for aid in self._get_random_agents():
            self._observations[aid], self._infos[aid] = self.agents[aid].reset()
        return (self._observations, self._infos)

    def step(self, action_dict):
        rew, terminated, truncated = ({}, {}, {})
        for aid, action in action_dict.items():
            self._observations[aid], rew[aid], terminated[aid], truncated[aid], self._infos[aid] = self.agents[aid].step(action)
            if terminated[aid]:
                self.terminateds.add(aid)
            if truncated[aid]:
                self.truncateds.add(aid)
        terminated['__all__'] = len(self.terminateds) == self.num_agents
        truncated['__all__'] = len(self.truncateds) == self.num_agents
        obs = {}
        infos = {}
        for aid in self._get_random_agents():
            if aid not in self._observations:
                self._observations[aid] = self.observation_space.sample()
                self._infos[aid] = {'fourty-two': 42}
            obs[aid] = self._observations.pop(aid)
            infos[aid] = self._infos.pop(aid)
        for aid in self._get_random_agents():
            rew[aid] = np.random.rand()
        return (obs, rew, terminated, truncated, infos)

    def _get_random_agents(self):
        num_observing_agents = np.random.randint(self.num_agents)
        aids = np.random.permutation(self.num_agents)[:num_observing_agents]
        return {aid for aid in aids if aid not in self.terminateds and aid not in self.truncateds}