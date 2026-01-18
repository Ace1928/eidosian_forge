import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _iter_inst(self):
    for inst in self.bytecode:
        if self._use_new_block(inst):
            self._guard_with_as(inst)
            self._start_new_block(inst)
        self._curblock.body.append(inst.offset)
        yield inst