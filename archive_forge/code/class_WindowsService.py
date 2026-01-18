import contextlib
import errno
import functools
import os
import signal
import sys
import time
from collections import namedtuple
from . import _common
from ._common import ENCODING
from ._common import ENCODING_ERRS
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import debug
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import parse_environ_block
from ._common import usage_percent
from ._compat import PY3
from ._compat import long
from ._compat import lru_cache
from ._compat import range
from ._compat import unicode
from ._psutil_windows import ABOVE_NORMAL_PRIORITY_CLASS
from ._psutil_windows import BELOW_NORMAL_PRIORITY_CLASS
from ._psutil_windows import HIGH_PRIORITY_CLASS
from ._psutil_windows import IDLE_PRIORITY_CLASS
from ._psutil_windows import NORMAL_PRIORITY_CLASS
from ._psutil_windows import REALTIME_PRIORITY_CLASS
class WindowsService:
    """Represents an installed Windows service."""

    def __init__(self, name, display_name):
        self._name = name
        self._display_name = display_name

    def __str__(self):
        details = '(name=%r, display_name=%r)' % (self._name, self._display_name)
        return '%s%s' % (self.__class__.__name__, details)

    def __repr__(self):
        return '<%s at %s>' % (self.__str__(), id(self))

    def __eq__(self, other):
        if not isinstance(other, WindowsService):
            return NotImplemented
        return self._name == other._name

    def __ne__(self, other):
        return not self == other

    def _query_config(self):
        with self._wrap_exceptions():
            display_name, binpath, username, start_type = cext.winservice_query_config(self._name)
        return dict(display_name=py2_strencode(display_name), binpath=py2_strencode(binpath), username=py2_strencode(username), start_type=py2_strencode(start_type))

    def _query_status(self):
        with self._wrap_exceptions():
            status, pid = cext.winservice_query_status(self._name)
        if pid == 0:
            pid = None
        return dict(status=status, pid=pid)

    @contextlib.contextmanager
    def _wrap_exceptions(self):
        """Ctx manager which translates bare OSError and WindowsError
        exceptions into NoSuchProcess and AccessDenied.
        """
        try:
            yield
        except OSError as err:
            if is_permission_err(err):
                msg = 'service %r is not querable (not enough privileges)' % self._name
                raise AccessDenied(pid=None, name=self._name, msg=msg)
            elif err.winerror in (cext.ERROR_INVALID_NAME, cext.ERROR_SERVICE_DOES_NOT_EXIST):
                msg = 'service %r does not exist' % self._name
                raise NoSuchProcess(pid=None, name=self._name, msg=msg)
            else:
                raise

    def name(self):
        """The service name. This string is how a service is referenced
        and can be passed to win_service_get() to get a new
        WindowsService instance.
        """
        return self._name

    def display_name(self):
        """The service display name. The value is cached when this class
        is instantiated.
        """
        return self._display_name

    def binpath(self):
        """The fully qualified path to the service binary/exe file as
        a string, including command line arguments.
        """
        return self._query_config()['binpath']

    def username(self):
        """The name of the user that owns this service."""
        return self._query_config()['username']

    def start_type(self):
        """A string which can either be "automatic", "manual" or
        "disabled".
        """
        return self._query_config()['start_type']

    def pid(self):
        """The process PID, if any, else None. This can be passed
        to Process class to control the service's process.
        """
        return self._query_status()['pid']

    def status(self):
        """Service status as a string."""
        return self._query_status()['status']

    def description(self):
        """Service long description."""
        return py2_strencode(cext.winservice_query_descr(self.name()))

    def as_dict(self):
        """Utility method retrieving all the information above as a
        dictionary.
        """
        d = self._query_config()
        d.update(self._query_status())
        d['name'] = self.name()
        d['display_name'] = self.display_name()
        d['description'] = self.description()
        return d