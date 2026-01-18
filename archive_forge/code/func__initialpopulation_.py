from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
def _initialpopulation_(self):
    for _ in range(Population.population):
        self.snakes.append(Snake(Population.hidden_node))