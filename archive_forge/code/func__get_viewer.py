from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
def _get_viewer(self, mode) -> Union['gym.envs.mujoco.mujoco_rendering.Viewer', 'gym.envs.mujoco.mujoco_rendering.RenderContextOffscreen']:
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
        if mode == 'human':
            from gym.envs.mujoco.mujoco_rendering import Viewer
            self.viewer = Viewer(self.model, self.data)
        elif mode in {'rgb_array', 'depth_array'}:
            from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen
            self.viewer = RenderContextOffscreen(self.model, self.data)
        else:
            raise AttributeError(f'Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}')
        self.viewer_setup()
        self._viewers[mode] = self.viewer
    return self.viewer