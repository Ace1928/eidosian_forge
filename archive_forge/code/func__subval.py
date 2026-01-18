import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
def _subval(self, xs, idx, x):
    return self.seq_type(*subvals(xs, [(idx, x)]))