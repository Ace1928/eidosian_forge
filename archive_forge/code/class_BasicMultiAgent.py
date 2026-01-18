import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
class BasicMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps."""
    metadata = {'render.modes': ['rgb_array']}
    render_mode = 'rgb_array'

    def __init__(self, num):
        super().__init__()
        self.agents = [MockEnv(25) for _ in range(num)]
        self._agent_ids = set(range(num))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        reset_results = [a.reset() for a in self.agents]
        return ({i: oi[0] for i, oi in enumerate(reset_results)}, {i: oi[1] for i, oi in enumerate(reset_results)})

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = ({}, {}, {}, {}, {})
        for i, action in action_dict.items():
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(action)
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        terminated['__all__'] = len(self.terminateds) == len(self.agents)
        truncated['__all__'] = len(self.truncateds) == len(self.agents)
        return (obs, rew, terminated, truncated, info)

    def render(self):
        return np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)