from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
class BaseMujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(self, model_path, frame_skip, observation_space: Space, render_mode: Optional[str]=None, width: int=DEFAULT_SIZE, height: int=DEFAULT_SIZE, camera_id: Optional[int]=None, camera_name: Optional[str]=None):
        if model_path.startswith('/'):
            self.fullpath = model_path
        else:
            self.fullpath = path.join(path.dirname(__file__), 'assets', model_path)
        if not path.exists(self.fullpath):
            raise OSError(f'File {self.fullpath} does not exist')
        self.width = width
        self.height = height
        self._initialize_simulation()
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}
        self.frame_skip = frame_skip
        self.viewer = None
        assert self.metadata['render_modes'] == ['human', 'rgb_array', 'depth_array'], self.metadata['render_modes']
        assert int(np.round(1.0 / self.dt)) == self.metadata['render_fps'], f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata['render_fps']}'
        self.observation_space = observation_space
        self._set_action_space()
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

    def _initialize_simulation(self):
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _reset_simulation(self):
        """
        Reset MuJoCo simulation data structures, mjModel and mjData.
        """
        raise NotImplementedError

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        raise NotImplementedError

    def render(self):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self._reset_simulation()
        ob = self.reset_model()
        if self.render_mode == 'human':
            self.render()
        return (ob, {})

    def set_state(self, qpos, qvel):
        """
        Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError('Action dimension mismatch')
        self._step_mujoco_simulation(ctrl, n_frames)

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def get_body_com(self, body_name):
        """Return the cartesian position of a body frame"""
        raise NotImplementedError

    def state_vector(self):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])