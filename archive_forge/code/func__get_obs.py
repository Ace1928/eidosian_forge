from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
def _get_obs(self):
    return np.concatenate([self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat])