import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
Checks if the next instruction after a SETUP_WITH is something other
        than a POP_TOP, if it is something else it'll be some sort of store
        which is not supported (this corresponds to `with CTXMGR as VAR(S)`).