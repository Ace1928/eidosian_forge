from typing import Optional
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict
@PublicAPI
class ParallelPettingZooEnv(MultiAgentEnv):

    def __init__(self, env):
        super().__init__()
        self.par_env = env
        self.par_env.reset()
        self._agent_ids = set(self.par_env.agents)
        self.observation_space = gym.spaces.Dict(self.par_env.observation_spaces)
        self.action_space = gym.spaces.Dict(self.par_env.action_spaces)

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return (obs, info or {})

    def step(self, action_dict):
        obss, rews, terminateds, truncateds, infos = self.par_env.step(action_dict)
        terminateds['__all__'] = all(terminateds.values())
        truncateds['__all__'] = all(truncateds.values())
        return (obss, rews, terminateds, truncateds, infos)

    def close(self):
        self.par_env.close()

    def render(self):
        return self.par_env.render(self.render_mode)

    @property
    def get_sub_environments(self):
        return self.par_env.unwrapped