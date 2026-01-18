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
class AbstractItemOrAction(dict):

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    @property
    def name(self):
        return self.__getitem__('name')

    @name.setter
    def name(self, value):
        self.__setitem__('name', value)

    @property
    def value(self):
        return self.__getitem__('value')

    @value.setter
    def value(self, value):
        self.__setitem__('value', value)

    def is_item(self):
        return self.get('type') == 'item'

    def is_action(self):
        return self.get('type') == 'action'