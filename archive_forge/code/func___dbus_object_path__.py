import sys
import logging
import threading
import traceback
import _dbus_bindings
from dbus import (
from dbus.decorators import method, signal
from dbus.exceptions import (
from dbus.lowlevel import ErrorMessage, MethodReturnMessage, MethodCallMessage
from dbus.proxies import LOCAL_PATH
from dbus._compat import is_py2
@property
def __dbus_object_path__(self):
    """The object-path at which this object is available.
        Access raises AttributeError if there is no object path, or more than
        one object path.

        Changed in 0.82.0: AttributeError can be raised.
        """
    if self._object_path is _MANY:
        raise AttributeError('Object %r has more than one object path: use Object.locations instead' % self)
    elif self._object_path is None:
        raise AttributeError('Object %r has no object path yet' % self)
    else:
        return self._object_path