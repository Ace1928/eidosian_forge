import copy
import pickle
import sys
import typing
import warnings
from types import FunctionType
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
class CannedArray(CannedObject):
    """A canned numpy array."""

    def __init__(self, obj):
        """Initialize the can."""
        from numpy import ascontiguousarray
        self.shape = obj.shape
        self.dtype = obj.dtype.descr if obj.dtype.fields else obj.dtype.str
        self.pickled = False
        if sum(obj.shape) == 0:
            self.pickled = True
        elif obj.dtype == 'O':
            self.pickled = True
        elif obj.dtype.fields and any((dt == 'O' for dt, sz in obj.dtype.fields.values())):
            self.pickled = True
        if self.pickled:
            self.buffers = [pickle.dumps(obj, PICKLE_PROTOCOL)]
        else:
            obj = ascontiguousarray(obj, dtype=None)
            self.buffers = [buffer(obj)]

    def get_object(self, g=None):
        """Get the object."""
        from numpy import frombuffer
        data = self.buffers[0]
        if self.pickled:
            return pickle.loads(data)
        return frombuffer(data, dtype=self.dtype).reshape(self.shape)