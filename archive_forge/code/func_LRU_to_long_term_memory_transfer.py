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
def LRU_to_long_term_memory_transfer(LRU_memory: LRUMemory, long_term_memory: LongTermMemory) -> None:
    """
    Transfers relevant information from LRU memory to long-term memory for learning and optimisation.

    Args:
        LRU_memory (LRUMemory): The LRU memory instance.
        long_term_memory (LongTermMemory): The long-term memory instance.
    """
    for key, value in LRU_memory.cache.items():
        board, move, score = value
        long_term_memory.store(board, move, score)