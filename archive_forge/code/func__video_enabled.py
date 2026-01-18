import os
from typing import Callable, Optional
import gym
from gym import logger
from gym.wrappers.monitoring import video_recorder
def _video_enabled(self):
    if self.step_trigger:
        return self.step_trigger(self.step_id)
    else:
        return self.episode_trigger(self.episode_id)