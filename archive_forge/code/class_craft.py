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
class craft(Enum):
    crafting_table = 0
    none = 1
    planks = 2
    stick = 3
    torch = 4