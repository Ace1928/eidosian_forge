from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
class GeneratorInfo(object):

    def __init__(self):
        self.yield_points = {}
        self.state_vars = []

    def get_yield_points(self):
        """
        Return an iterable of YieldPoint instances.
        """
        return self.yield_points.values()