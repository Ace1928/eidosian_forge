import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class UserModeHandle(Handle):
    """
    Base class for non-kernel handles. Generally this means they are closed
    by special Win32 API functions instead of CloseHandle() and some standard
    operations (synchronizing, duplicating, inheritance) are not supported.

    @type _TYPE: C type
    @cvar _TYPE: C type to translate this handle to.
        Subclasses should override this.
        Defaults to L{HANDLE}.
    """
    _TYPE = HANDLE

    def _close(self):
        raise NotImplementedError()

    @property
    def _as_parameter_(self):
        return self._TYPE(self.value)

    @staticmethod
    def from_param(value):
        return self._TYPE(self.value)

    @property
    def inherit(self):
        return False

    @property
    def protectFromClose(self):
        return False

    def dup(self):
        raise NotImplementedError()

    def wait(self, dwMilliseconds=None):
        raise NotImplementedError()