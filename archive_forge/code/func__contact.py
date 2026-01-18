import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def _contact(self, contact, begin):
    tile = None
    obj = None
    u1 = contact.fixtureA.body.userData
    u2 = contact.fixtureB.body.userData
    if u1 and 'road_friction' in u1.__dict__:
        tile = u1
        obj = u2
    if u2 and 'road_friction' in u2.__dict__:
        tile = u2
        obj = u1
    if not tile:
        return
    tile.color[:] = self.env.road_color
    if not obj or 'tiles' not in obj.__dict__:
        return
    if begin:
        obj.tiles.add(tile)
        if not tile.road_visited:
            tile.road_visited = True
            self.env.reward += 1000.0 / len(self.env.track)
            self.env.tile_visited_count += 1
            if tile.idx == 0 and self.env.tile_visited_count / len(self.env.track) > self.lap_complete_percent:
                self.env.new_lap = True
    else:
        obj.tiles.remove(tile)