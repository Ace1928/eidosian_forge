from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
class KeysViewHDF5(KeysView):

    def __str__(self):
        return '<KeysViewHDF5 {}>'.format(list(self))

    def __reversed__(self):
        yield from reversed(self._mapping)
    __repr__ = __str__