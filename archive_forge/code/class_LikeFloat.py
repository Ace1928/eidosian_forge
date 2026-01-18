from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn
class LikeFloat(object):

    def __init__(self, n):
        self.n = n

    def __lt__(self, other):
        return self.n < other

    def __gt__(self, other):
        return self.n > other