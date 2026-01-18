import math
from typing import TYPE_CHECKING, List, Optional
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle
def _generate_clouds(self):
    self.cloud_poly = []
    for i in range(TERRAIN_LENGTH // 20):
        x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
        y = VIEWPORT_H / SCALE * 3 / 4
        poly = [(x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP), y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP)) for a in range(5)]
        x1 = min((p[0] for p in poly))
        x2 = max((p[0] for p in poly))
        self.cloud_poly.append((poly, x1, x2))