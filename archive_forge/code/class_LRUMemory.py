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
class LRUMemory:
    """
    Implements a Least Recently Used (LRU) memory cache to store game states, moves, and scores.
    """

    @StandardDecorator()
    def __init__(self, capacity: int=50):
        self.capacity = capacity
        self.cache: Dict[Tuple[str, int], np.ndarray] = collections.OrderedDict()

    @StandardDecorator
    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the LRU memory, evicting the least recently used item if necessary.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.
        """
        key = (move, score)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = board
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)