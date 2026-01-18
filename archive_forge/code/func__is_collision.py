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
def _is_collision(self, node: Tuple[int, int]) -> bool:
    return node in self.body or node in self.obstacles