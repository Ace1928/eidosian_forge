import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _start_new_block(self, inst):
    self._curblock = CFBlock(inst.offset)
    self.blocks[inst.offset] = self._curblock
    self.blockseq.append(inst.offset)