from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
def _initialize_simulation(self):
    self.model = mujoco.MjModel.from_xml_path(self.fullpath)
    self.model.vis.global_.offwidth = self.width
    self.model.vis.global_.offheight = self.height
    self.data = mujoco.MjData(self.model)