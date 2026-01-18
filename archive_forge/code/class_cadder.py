import os
import math
import dill as pickle
class cadder(object):

    def __init__(self, augend):
        self.augend = augend
        self.zero = [0]

    def __call__(self, addend):
        return addend + self.augend + self.zero[0]