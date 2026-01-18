import pygame
import pygame_gui
import numpy as np
from collections import deque
from typing import List, Tuple, Deque, Dict, Any, Optional
import threading
import time
import random
import math
import asyncio
import os
import logging
import sys
import aiofiles
from functools import lru_cache as LRUCache
import aiohttp
import json
import cachetools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.utils.data.distributed as distributed
import torch.distributions as distributions
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.cuda as cuda  # Added for potential GPU acceleration
import torch.backends.cudnn as cudnn  # Added for optimizing deep learning computations on CUDA
import logging  # For detailed logging of operations and errors
import hashlib  # For generating unique identifiers for nodes
import bisect  # For maintaining sorted lists
import gc  # For explicit garbage collection if necessary
class Perception:

    def __init__(self, grid):
        """
        Initialize the Perception class which is responsible for evaluating and updating the game's perception of the environment.

        Args:
            grid (Grid): The game grid which contains all the necessary information about the game state.

        Attributes:
            grid (Grid): Stores the reference to the game grid.
            perception_cache (LRUCache): Caches the results of complex perception calculations for quick retrieval.
            perception_lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations during updates.
        """
        self.grid = grid
        self.perception_cache = LRUCache(maxsize=1024)
        self.perception_lock = asyncio.Lock()

    async def update_perception(self, snake, fruit):
        """
        Asynchronously updates the perception of the environment based on the current state of the snake and the fruit.

        Args:
            snake (Snake): The current state of the snake.
            fruit (Fruit): The current state of the fruit.

        Returns:
            None
        """
        async with self.perception_lock:
            result = self._compute_perception(snake, fruit)
            self.perception_cache[hash((snake, fruit))] = result

    @staticmethod
    def _compute_perception(snake, fruit):
        """
        Compute the perception details such as potential collisions, pathfinding evaluations, and strategic positioning.

        Args:
            snake (Snake): The current state of the snake.
            fruit (Fruit): The current state of the fruit.

        Returns:
            Dict[str, Any]: A dictionary containing detailed perception metrics.
        """
        path = np.linalg.norm(np.array(snake.get_head_position()) - np.array(fruit.get_position()))
        return {'path_length': path, 'collision_risk': Perception._evaluate_collision_risk(snake, path), 'strategic_advantage': Perception._calculate_strategic_advantage(snake, fruit)}

    @staticmethod
    def _evaluate_collision_risk(snake, path_length):
        """
        Evaluate the risk of collision based on the path length and snake's current trajectory.

        Args:
            snake (Snake): The current state of the snake.
            path_length (float): The computed path length from the snake's head to the fruit.

        Returns:
            float: A risk factor indicating the likelihood of collision.
        """
        return path_length / (1 + len(snake.get_segments()))

    @staticmethod
    def _calculate_strategic_advantage(snake, fruit):
        """
        Calculate the strategic advantage of moving towards the fruit based on current game metrics.

        Args:
            snake (Snake): The current state of the snake.
            fruit (Fruit): The current state of the fruit.

        Returns:
            float: A score representing the strategic advantage.
        """
        return np.random.random()