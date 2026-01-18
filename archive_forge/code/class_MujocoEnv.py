from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
class MujocoEnv(BaseMujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(self, model_path, frame_skip, observation_space: Space, render_mode: Optional[str]=None, width: int=DEFAULT_SIZE, height: int=DEFAULT_SIZE, camera_id: Optional[int]=None, camera_name: Optional[str]=None):
        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(f'{MUJOCO_IMPORT_ERROR}. (HINT: you need to install mujoco)')
        super().__init__(model_path, frame_skip, observation_space, render_mode, width, height, camera_id, camera_name)

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(f'You are calling render method without specifying any render mode. You can specify the render_mode at initialization, e.g. gym("{self.spec.id}", render_mode="rgb_array")')
            return
        if self.render_mode in {'rgb_array', 'depth_array'}:
            camera_id = self.camera_id
            camera_name = self.camera_name
            if camera_id is not None and camera_name is not None:
                raise ValueError('Both `camera_id` and `camera_name` cannot be specified at the same time.')
            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'
            if camera_id is None:
                camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                self._get_viewer(self.render_mode).render(camera_id=camera_id)
        if self.render_mode == 'rgb_array':
            data = self._get_viewer(self.render_mode).read_pixels(depth=False)
            return data[::-1, :, :]
        elif self.render_mode == 'depth_array':
            self._get_viewer(self.render_mode).render()
            data = self._get_viewer(self.render_mode).read_pixels(depth=True)[1]
            return data[::-1, :]
        elif self.render_mode == 'human':
            self._get_viewer(self.render_mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()

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

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos