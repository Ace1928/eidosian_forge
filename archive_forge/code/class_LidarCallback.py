import math
from typing import TYPE_CHECKING, List, Optional
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle
class LidarCallback(Box2D.b2.rayCastCallback):

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.filterData.categoryBits & 1 == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        return fraction