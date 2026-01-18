import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle, clip=True):
    poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
    poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly]
    if not clip or any((-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM and -MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM for coord in poly)):
        gfxdraw.aapolygon(self.surf, poly, color)
        gfxdraw.filled_polygon(self.surf, poly, color)