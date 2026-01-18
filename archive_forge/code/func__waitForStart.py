import errno
import os
import pwd
import sys
import traceback
from twisted import copyright, logger
from twisted.application import app, service
from twisted.internet.interfaces import IReactorDaemonize
from twisted.python import log, logfile, usage
from twisted.python.runtime import platformType
from twisted.python.util import gidFromString, switchUID, uidFromString, untilConcludes
def _waitForStart(self, readPipe: int) -> int:
    """
        Wait for the daemonization success.

        @param readPipe: file descriptor to read start information from.
        @type readPipe: C{int}

        @return: code to be passed to C{os._exit}: 0 for success, 1 for error.
        @rtype: C{int}
        """
    data = untilConcludes(os.read, readPipe, 100)
    dataRepr = repr(data[2:])
    if data != b'0':
        msg = 'An error has occurred: {}\nPlease look at log file for more information.\n'.format(dataRepr)
        untilConcludes(sys.__stderr__.write, msg)
        return 1
    return 0