import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _render_indicators(self, W, H):
    s = W / 40.0
    h = H / 40.0
    color = (0, 0, 0)
    polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
    pygame.draw.polygon(self.surf, color=color, points=polygon)

    def vertical_ind(place, val):
        return [(place * s, H - (h + h * val)), ((place + 1) * s, H - (h + h * val)), ((place + 1) * s, H - h), ((place + 0) * s, H - h)]

    def horiz_ind(place, val):
        return [((place + 0) * s, H - 4 * h), ((place + val) * s, H - 4 * h), ((place + val) * s, H - 2 * h), ((place + 0) * s, H - 2 * h)]
    assert self.car is not None
    true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))

    def render_if_min(value, points, color):
        if abs(value) > 0.0001:
            pygame.draw.polygon(self.surf, points=points, color=color)
    render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
    render_if_min(self.car.wheels[0].omega, vertical_ind(7, 0.01 * self.car.wheels[0].omega), (0, 0, 255))
    render_if_min(self.car.wheels[1].omega, vertical_ind(8, 0.01 * self.car.wheels[1].omega), (0, 0, 255))
    render_if_min(self.car.wheels[2].omega, vertical_ind(9, 0.01 * self.car.wheels[2].omega), (51, 0, 255))
    render_if_min(self.car.wheels[3].omega, vertical_ind(10, 0.01 * self.car.wheels[3].omega), (51, 0, 255))
    render_if_min(self.car.wheels[0].joint.angle, horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle), (0, 255, 0))
    render_if_min(self.car.hull.angularVelocity, horiz_ind(30, -0.8 * self.car.hull.angularVelocity), (255, 0, 0))