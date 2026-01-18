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
class _VariantSignature(object):
    """A fake method signature which, when iterated, yields an endless stream
    of 'v' characters representing variants (handy with zip()).

    It has no string representation.
    """

    def __iter__(self):
        """Return self."""
        return self

    def __next__(self):
        """Return 'v' whenever called."""
        return 'v'