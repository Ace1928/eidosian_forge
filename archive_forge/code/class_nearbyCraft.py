import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
class nearbyCraft(Enum):
    furnace = 0
    iron_axe = 1
    iron_pickaxe = 2
    none = 3
    stone_axe = 4
    stone_pickaxe = 5
    wooden_axe = 6
    wooden_pickaxe = 7