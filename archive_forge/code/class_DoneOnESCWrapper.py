from typing import List, Optional, Sequence
import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
class DoneOnESCWrapper(gym.Wrapper):
    """
    Use the "ESC" action of the MineRL 1.0.0 to end
    an episode (if 1, step will return done=True)
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_over = False

    def reset(self):
        self.episode_over = False
        return self.env.reset()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError('Expected `reset` after episode terminated, not `step`.')
        observation, reward, done, info = self.env.step(action)
        done = done or bool(action['ESC'])
        self.episode_over = done
        return (observation, reward, done, info)