from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def _scalar_mul(self, x, a):
    return x * a