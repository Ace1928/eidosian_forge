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
class DummyDataLoader:

    def __init__(self, data, items_to_add):
        self.data = data
        self.items_to_add = items_to_add

    def batch_iter(self, *args, **kwargs):
        for item in self.items_to_add:
            for slice_ in self.data[item]:
                yield slice_