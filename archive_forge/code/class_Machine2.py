import dill
from functools import partial
import warnings
class Machine2(object):

    def __init__(self):
        self.go = partial(self.member, self)

    def member(self, model):
        pass