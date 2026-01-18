import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class ThreadHandle(Handle):
    """
    Win32 thread handle.

    @type dwAccess: int
    @ivar dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenThread}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{THREAD_ALL_ACCESS}.

    @see: L{Handle}
    """

    def __init__(self, aHandle=None, bOwnership=True, dwAccess=THREAD_ALL_ACCESS):
        """
        @type  aHandle: int
        @param aHandle: Win32 handle value.

        @type  bOwnership: bool
        @param bOwnership:
           C{True} if we own the handle and we need to close it.
           C{False} if someone else will be calling L{CloseHandle}.

        @type  dwAccess: int
        @param dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenThread}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{THREAD_ALL_ACCESS}.
        """
        super(ThreadHandle, self).__init__(aHandle, bOwnership)
        self.dwAccess = dwAccess
        if aHandle is not None and dwAccess is None:
            msg = 'Missing access flags for thread handle: %x' % aHandle
            raise TypeError(msg)

    def get_tid(self):
        """
        @rtype:  int
        @return: Thread global ID.
        """
        return GetThreadId(self.value)