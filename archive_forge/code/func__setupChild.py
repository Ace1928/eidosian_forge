from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def _setupChild(self, masterfd, slavefd):
    """
        Set up child process after C{fork()} but before C{exec()}.

        This involves:

            - closing C{masterfd}, since it is not used in the subprocess

            - creating a new session with C{os.setsid}

            - changing the controlling terminal of the process (and the new
              session) to point at C{slavefd}

            - duplicating C{slavefd} to standard input, output, and error

            - closing all other open file descriptors (according to
              L{_listOpenFDs})

            - re-setting all signal handlers to C{SIG_DFL}

        @param masterfd: The master end of a PTY file descriptors opened with
            C{openpty}.
        @type masterfd: L{int}

        @param slavefd: The slave end of a PTY opened with C{openpty}.
        @type slavefd: L{int}
        """
    os.close(masterfd)
    os.setsid()
    fcntl.ioctl(slavefd, termios.TIOCSCTTY, '')
    for fd in range(3):
        if fd != slavefd:
            os.close(fd)
    os.dup2(slavefd, 0)
    os.dup2(slavefd, 1)
    os.dup2(slavefd, 2)
    for fd in _listOpenFDs():
        if fd > 2:
            try:
                os.close(fd)
            except BaseException:
                pass
    self._resetSignalDisposition()