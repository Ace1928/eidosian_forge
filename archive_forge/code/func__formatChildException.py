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
def _formatChildException(self, exception):
    """
        Format the C{exception} in preparation for writing to the
        status pipe.  This does the right thing on Python 2 if the
        exception's message is Unicode, and in all cases limits the
        length of the message afte* encoding to 100 bytes.

        This means the returned message may be truncated in the middle
        of a unicode escape.

        @type exception: L{Exception}
        @param exception: The exception to format.

        @return: The formatted message, suitable for writing to the
            status pipe.
        @rtype: L{bytes}
        """
    exceptionLine = traceback.format_exception_only(exception.__class__, exception)[-1]
    formattedMessage = f'1 {exceptionLine.strip()}'
    formattedMessage = formattedMessage.encode('ascii', 'backslashreplace')
    return formattedMessage[:100]