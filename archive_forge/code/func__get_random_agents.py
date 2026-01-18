import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
def _get_random_agents(self):
    num_observing_agents = np.random.randint(self.num_agents)
    aids = np.random.permutation(self.num_agents)[:num_observing_agents]
    return {aid for aid in aids if aid not in self.terminateds and aid not in self.truncateds}