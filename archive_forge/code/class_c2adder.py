import os
import math
import dill as pickle
class c2adder:

    def __init__(self, augend):
        self.augend = augend
        self.zero = [0]

    def __call__(self, addend):
        return addend + self.augend + self.zero[0]