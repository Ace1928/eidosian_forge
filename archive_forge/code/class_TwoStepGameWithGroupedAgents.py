from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
class TwoStepGameWithGroupedAgents(MultiAgentEnv):

    def __init__(self, env_config):
        super().__init__()
        env = TwoStepGame(env_config)
        tuple_obs_space = Tuple([env.observation_space, env.observation_space])
        tuple_act_space = Tuple([env.action_space, env.action_space])
        self.env = env.with_agent_groups(groups={'agents': [0, 1]}, obs_space=tuple_obs_space, act_space=tuple_act_space)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = {'agents'}
        self._skip_env_checking = True

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)