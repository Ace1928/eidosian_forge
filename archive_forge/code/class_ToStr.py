from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn
class ToStr(object):

    def __init__(self, n):
        self.n = str(n)

    def __str__(self):
        return self.n