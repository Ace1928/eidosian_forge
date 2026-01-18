import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
@property
def H(self):
    return self.getH()