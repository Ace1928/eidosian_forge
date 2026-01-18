import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _create_image_array(self, screen, size):
    scaled_screen = pygame.transform.smoothscale(screen, size)
    return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))