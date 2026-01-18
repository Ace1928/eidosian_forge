import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
class CFBlock(object):

    def __init__(self, offset):
        self.offset = offset
        self.body = []
        self.outgoing_jumps = {}
        self.incoming_jumps = {}
        self.terminating = False

    def __repr__(self):
        args = (self.offset, sorted(self.outgoing_jumps), sorted(self.incoming_jumps))
        return 'block(offset:%d, outgoing: %s, incoming: %s)' % args

    def __iter__(self):
        return iter(self.body)