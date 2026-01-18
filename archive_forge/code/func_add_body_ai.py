from pygame.math import Vector2
from Fruit import Fruit
from NN import NeuralNework
import pickle
def add_body_ai(self):
    last_indx = len(self.body) - 1
    tail = self.body[-1]
    before_last = self.body[-2]
    if tail.x == before_last.x:
        if tail.y < before_last.y:
            self.body.append(Vector2(tail.x, tail.y - 1))
        else:
            self.body.append(Vector2(tail.x, tail.y + 1))
    elif tail.y == before_last.y:
        if tail.x < before_last.x:
            self.body.append(Vector2(tail.x - 1, tail.y))
        else:
            self.body.append(Vector2(tail.x + 1, tail.y))