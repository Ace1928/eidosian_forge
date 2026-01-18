import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _op_ABSOLUTE_JUMP_IF(self, inst):
    self.jump(inst.get_jump_target())
    self.jump(inst.next)
    self._force_new_block = True