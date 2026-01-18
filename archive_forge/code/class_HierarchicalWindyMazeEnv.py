import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import logging
import random
from ray.rllib.env import MultiAgentEnv
class HierarchicalWindyMazeEnv(MultiAgentEnv):

    def __init__(self, env_config):
        super().__init__()
        self._skip_env_checking = True
        self.flat_env = WindyMazeEnv(env_config)

    def reset(self, *, seed=None, options=None):
        self.cur_obs, infos = self.flat_env.reset()
        self.current_goal = None
        self.steps_remaining_at_level = None
        self.num_high_level_steps = 0
        self.low_level_agent_id = 'low_level_{}'.format(self.num_high_level_steps)
        return ({'high_level_agent': self.cur_obs}, {'high_level_agent': infos})

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if 'high_level_agent' in action_dict:
            return self._high_level_step(action_dict['high_level_agent'])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action):
        logger.debug('High level agent sets goal')
        self.current_goal = action
        self.steps_remaining_at_level = 25
        self.num_high_level_steps += 1
        self.low_level_agent_id = 'low_level_{}'.format(self.num_high_level_steps)
        obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
        rew = {self.low_level_agent_id: 0}
        done = truncated = {'__all__': False}
        return (obs, rew, done, truncated, {})

    def _low_level_step(self, action):
        logger.debug('Low level agent step {}'.format(action))
        self.steps_remaining_at_level -= 1
        cur_pos = tuple(self.cur_obs[0])
        goal_pos = self.flat_env._get_new_pos(cur_pos, self.current_goal)
        f_obs, f_rew, f_terminated, f_truncated, info = self.flat_env.step(action)
        new_pos = tuple(f_obs[0])
        self.cur_obs = f_obs
        obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
        if new_pos != cur_pos:
            if new_pos == goal_pos:
                rew = {self.low_level_agent_id: 1}
            else:
                rew = {self.low_level_agent_id: -1}
        else:
            rew = {self.low_level_agent_id: 0}
        terminated = {'__all__': False}
        truncated = {'__all__': False}
        if f_terminated or f_truncated:
            terminated['__all__'] = f_terminated
            truncated['__all__'] = f_truncated
            logger.debug('high level final reward {}'.format(f_rew))
            rew['high_level_agent'] = f_rew
            obs['high_level_agent'] = f_obs
        elif self.steps_remaining_at_level == 0:
            terminated[self.low_level_agent_id] = True
            truncated[self.low_level_agent_id] = False
            rew['high_level_agent'] = 0
            obs['high_level_agent'] = f_obs
        return (obs, rew, terminated, truncated, {self.low_level_agent_id: info})