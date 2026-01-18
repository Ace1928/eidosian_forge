import pygame as pg
import sys
from random import randint, seed
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Deque, Set, Optional
from heapq import heappush, heappop
import numpy as np
import math
from queue import PriorityQueue
import logging
class Fruit:

    def __init__(self):
        """
        Initializes a Fruit object with a position on the game window.
        """
        self.position: Tuple[int, int] = (0, 0)
        self.relocate()
        logging.info(f'Fruit placed at {self.position}')

    def draw(self) -> None:
        """
        Draw the fruit on the game window using a fixed color (red) and block size.
        """
        pg.draw.rect(window, (255, 0, 0), (*self.position, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f'Fruit drawn at {self.position}')

    def relocate(self, exclude=None) -> None:
        """
        Relocate the fruit to a random position within the game boundaries that is not occupied.
        Ensures the fruit does not spawn inside the snake's body.
        """
        if exclude is None:
            exclude = []
        while True:
            new_x: int = randint(0, SCREEN_WIDTH // BLOCK_SIZE - 1) * BLOCK_SIZE
            new_y: int = randint(0, SCREEN_HEIGHT // BLOCK_SIZE - 1) * BLOCK_SIZE
            new_position: Tuple[int, int] = (new_x, new_y)
            if new_position not in exclude:
                self.position = new_position
                break
        logging.info(f'Fruit relocated to {self.position}')