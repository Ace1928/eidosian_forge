import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def _rank_all(self) -> None:
    """
        Ranks all data in the memory based on relevance.
        """
    for key in self.memory:
        self.memory[key] = (self.memory[key][0], self.memory[key][1], self.memory[key][2], self._rank(key))
        ranked_memory = sorted(self.memory.items(), key=lambda x: x[1][3], reverse=True)
        self.memory = dict(ranked_memory)
    return None