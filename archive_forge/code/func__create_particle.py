import math
import Box2D
import numpy as np
from gym.error import DependencyNotInstalled
def _create_particle(self, point1, point2, grass):

    class Particle:
        pass
    p = Particle()
    p.color = WHEEL_COLOR if not grass else MUD_COLOR
    p.ttl = 1
    p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
    p.grass = grass
    self.particles.append(p)
    while len(self.particles) > 30:
        self.particles.pop(0)
    return p