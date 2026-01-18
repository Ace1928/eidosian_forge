from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
class MuJocoPyEnv(BaseMujocoEnv):

    def __init__(self, model_path: str, frame_skip: int, observation_space: Space, render_mode: Optional[str]=None, width: int=DEFAULT_SIZE, height: int=DEFAULT_SIZE, camera_id: Optional[int]=None, camera_name: Optional[str]=None):
        if MUJOCO_PY_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(f'{MUJOCO_PY_IMPORT_ERROR}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)')
        logger.warn('This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works).')
        super().__init__(model_path, frame_skip, observation_space, render_mode, width, height, camera_id, camera_name)

    def _initialize_simulation(self):
        self.model = mujoco_py.load_model_from_path(self.fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

    def _reset_simulation(self):
        self.sim.reset()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        state = self.sim.get_state()
        state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)
        self.sim.set_state(state)
        self.sim.forward()

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(f'You are calling render method without specifying any render mode. You can specify the render_mode at initialization, e.g. gym("{self.spec.id}", render_mode="rgb_array")')
            return
        width, height = (self.width, self.height)
        camera_name, camera_id = (self.camera_name, self.camera_id)
        if self.render_mode in {'rgb_array', 'depth_array'}:
            if camera_id is not None and camera_name is not None:
                raise ValueError('Both `camera_id` and `camera_name` cannot be specified at the same time.')
            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'
            if camera_id is None and camera_name in self.model._camera_name2id:
                if camera_name in self.model._camera_name2id:
                    camera_id = self.model.camera_name2id(camera_name)
                self._get_viewer(self.render_mode).render(width, height, camera_id=camera_id)
        if self.render_mode == 'rgb_array':
            data = self._get_viewer(self.render_mode).read_pixels(width, height, depth=False)
            return data[::-1, :, :]
        elif self.render_mode == 'depth_array':
            self._get_viewer(self.render_mode).render(width, height)
            data = self._get_viewer(self.render_mode).read_pixels(width, height, depth=True)[1]
            return data[::-1, :]
        elif self.render_mode == 'human':
            self._get_viewer(self.render_mode).render()

    def _get_viewer(self, mode) -> Union['mujoco_py.MjViewer', 'mujoco_py.MjRenderContextOffscreen']:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode in {'rgb_array', 'depth_array'}:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            else:
                raise AttributeError(f'Unknown mode: {mode}, expected modes: {self.metadata['render_modes']}')
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)