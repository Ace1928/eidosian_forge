from Algorithm import Algorithm
from Snake import Snake
import math
import random
from Utility import Node
from Constants import NO_OF_CELLS, BANNER_HEIGHT, USER_SEED
import numpy as np
class Population:
    population = 300
    hidden_node = 8

    def __init__(self):
        self.snakes = []
        self.saved_snakes = []

    def _initialpopulation_(self):
        for _ in range(Population.population):
            self.snakes.append(Snake(Population.hidden_node))

    def remove(self, snake):
        self.saved_snakes.append(snake)
        self.snakes.remove(snake)