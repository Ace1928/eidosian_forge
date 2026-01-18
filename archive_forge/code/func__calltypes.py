from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
@property
def _calltypes(self):
    return self._lowerer.fndesc.calltypes