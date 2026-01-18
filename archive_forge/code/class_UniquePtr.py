import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class UniquePtr(object):
    """Wrapper around single-ownership C-API objects that handles deletion."""
    __slots__ = ['_obj', 'deleter', 'name', 'type_name']

    def __init__(self, name, obj, deleter):
        self._obj = obj
        self.name = name
        self.deleter = deleter
        self.type_name = str(type(obj))

    @contextlib.contextmanager
    def get(self):
        """Yields the managed C-API Object, guaranteeing aliveness.

    This is a context manager. Inside the context the C-API object is
    guaranteed to be alive.

    Raises:
      AlreadyGarbageCollectedError: if the object is already deleted.
    """
        if self._obj is None:
            raise AlreadyGarbageCollectedError(self.name, self.type_name)
        yield self._obj

    def __del__(self):
        obj = self._obj
        if obj is not None:
            self._obj = None
            self.deleter(obj)