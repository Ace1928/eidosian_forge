import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math
from queue import PriorityQueue
class AIController:
    """
    AI Controller that acts as the "player" controlling the snake using pathfinding.
    """

    def __init__(self, snake: Snake, pathfinding: Pathfinding) -> None:
        self.snake = snake
        self.pathfinding = pathfinding

    def make_decision(self) -> None:
        """
        Makes decisions for the snake's next move based on pathfinding.
        """
        self.snake.move()