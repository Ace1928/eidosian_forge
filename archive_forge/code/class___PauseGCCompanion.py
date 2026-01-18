import gc
from pyomo.common.multithread import MultiThreadWrapper
class __PauseGCCompanion(object):

    def __init__(self):
        self._stack_depth = 0