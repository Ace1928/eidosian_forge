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
class ChainInfo:

    def __init__(self, chain, reward, env_name, trajectory_name, id_, length, time_indexes):
        self.chain = chain
        self.reward = reward
        self.env = env_name
        self.trajectory_name = trajectory_name
        self.id = id_
        self.length = length
        self.time_indexes = time_indexes

    def __str__(self):
        return str(self.reward) + '\n' + str(self.chain)