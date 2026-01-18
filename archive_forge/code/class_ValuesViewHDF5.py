from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
class ValuesViewHDF5(ValuesView):
    """
        Wraps e.g. a Group or AttributeManager to provide a value view.

        Note that __contains__ will have poor performance as it has
        to scan all the links or attributes.
    """

    def __contains__(self, value):
        with phil:
            for key in self._mapping:
                if value == self._mapping.get(key):
                    return True
            return False

    def __iter__(self):
        with phil:
            for key in self._mapping:
                yield self._mapping.get(key)

    def __reversed__(self):
        with phil:
            for key in reversed(self._mapping):
                yield self._mapping.get(key)