from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
class TwoStepGame(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        super().__init__()
        self.action_space = Discrete(2)
        self.state = None
        self.agent_1 = 0
        self.agent_2 = 1
        self._skip_env_checking = True
        self.actions_are_logits = env_config.get('actions_are_logits', False)
        self.one_hot_state_encoding = env_config.get('one_hot_state_encoding', False)
        self.with_state = env_config.get('separate_state_space', False)
        self._agent_ids = {0, 1}
        if not self.one_hot_state_encoding:
            self.observation_space = Discrete(6)
            self.with_state = False
        elif self.with_state:
            self.observation_space = Dict({'obs': MultiDiscrete([2, 2, 2, 3]), ENV_STATE: MultiDiscrete([2, 2, 2])})
        else:
            self.observation_space = MultiDiscrete([2, 2, 2, 3])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([1, 0, 0])
        return (self._obs(), {})

    def step(self, action_dict):
        if self.actions_are_logits:
            action_dict = {k: np.random.choice([0, 1], p=v) for k, v in action_dict.items()}
        state_index = np.flatnonzero(self.state)
        if state_index == 0:
            action = action_dict[self.agent_1]
            assert action in [0, 1], action
            if action == 0:
                self.state = np.array([0, 1, 0])
            else:
                self.state = np.array([0, 0, 1])
            global_rew = 0
            terminated = False
        elif state_index == 1:
            global_rew = 7
            terminated = True
        else:
            if action_dict[self.agent_1] == 0 and action_dict[self.agent_2] == 0:
                global_rew = 0
            elif action_dict[self.agent_1] == 1 and action_dict[self.agent_2] == 1:
                global_rew = 8
            else:
                global_rew = 1
            terminated = True
        rewards = {self.agent_1: global_rew / 2.0, self.agent_2: global_rew / 2.0}
        obs = self._obs()
        terminateds = {'__all__': terminated}
        truncateds = {'__all__': False}
        infos = {self.agent_1: {'done': terminateds['__all__']}, self.agent_2: {'done': terminateds['__all__']}}
        return (obs, rewards, terminateds, truncateds, infos)

    def _obs(self):
        if self.with_state:
            return {self.agent_1: {'obs': self.agent_1_obs(), ENV_STATE: self.state}, self.agent_2: {'obs': self.agent_2_obs(), ENV_STATE: self.state}}
        else:
            return {self.agent_1: self.agent_1_obs(), self.agent_2: self.agent_2_obs()}

    def agent_1_obs(self):
        if self.one_hot_state_encoding:
            return np.concatenate([self.state, [1]])
        else:
            return np.flatnonzero(self.state)[0]

    def agent_2_obs(self):
        if self.one_hot_state_encoding:
            return np.concatenate([self.state, [2]])
        else:
            return np.flatnonzero(self.state)[0] + 3