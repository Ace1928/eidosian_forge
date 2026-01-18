import math
from typing import TYPE_CHECKING, List, Optional
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle
class BipedalWalkerHardcore:

    def __init__(self):
        raise error.Error('Error initializing BipedalWalkerHardcore Environment.\nCurrently, we do not support initializing this mode of environment by calling the class directly.\nTo use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\ngym.make("BipedalWalker-v3", hardcore=True)')