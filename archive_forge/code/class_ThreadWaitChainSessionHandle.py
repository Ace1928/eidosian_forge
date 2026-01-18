from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ThreadWaitChainSessionHandle(Handle):
    """
    Thread wait chain session handle.

    Returned by L{OpenThreadWaitChainSession}.

    @see: L{Handle}
    """

    def __init__(self, aHandle=None):
        """
        @type  aHandle: int
        @param aHandle: Win32 handle value.
        """
        super(ThreadWaitChainSessionHandle, self).__init__(aHandle, bOwnership=True)

    def _close(self):
        if self.value is None:
            raise ValueError('Handle was already closed!')
        CloseThreadWaitChainSession(self.value)

    def dup(self):
        raise NotImplementedError()

    def wait(self, dwMilliseconds=None):
        raise NotImplementedError()

    @property
    def inherit(self):
        return False

    @property
    def protectFromClose(self):
        return False