import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
class TrieNode(object):

    def __init__(self):
        super().__init__()
        self.num_leaves = 0
        self.children = {}
        self.hp_name = None

    def is_leaf(self):
        return len(self.children) == 0