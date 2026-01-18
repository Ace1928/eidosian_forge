from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class OperationFailed(FSError):
    """A specific operation failed."""
    default_message = 'operation failed, {details}'

    def __init__(self, path=None, exc=None, msg=None):
        self.path = path
        self.exc = exc
        self.details = '' if exc is None else text_type(exc)
        self.errno = getattr(exc, 'errno', None)
        super(OperationFailed, self).__init__(msg=msg)

    def __reduce__(self):
        return (type(self), (self.path, self.exc, self._msg))