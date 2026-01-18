from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
@toolz.curry
class NestedCurried(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @toolz.curry
    def f2(self, a, b):
        return self.x + self.y + a + b

    def g2(self):
        pass

    def __reduce__(self):
        """Allow us to serialize instances of NestedCurried"""
        return (GlobalCurried.NestedCurried, (self.x, self.y))