from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
class RemoteAmpError(AmpError):
    """
    This error indicates that something went wrong on the remote end of the
    connection, and the error was serialized and transmitted to you.
    """

    def __init__(self, errorCode, description, fatal=False, local=None):
        """Create a remote error with an error code and description.

        @param errorCode: the AMP error code of this error.
        @type errorCode: C{bytes}

        @param description: some text to show to the user.
        @type description: C{str}

        @param fatal: a boolean, true if this error should terminate the
        connection.

        @param local: a local Failure, if one exists.
        """
        if local:
            localwhat = ' (local)'
            othertb = local.getBriefTraceback()
        else:
            localwhat = ''
            othertb = ''
        errorCodeForMessage = ''.join((f'\\x{c:2x}' if c >= 128 else chr(c) for c in errorCode))
        if othertb:
            message = 'Code<{}>{}: {}\n{}'.format(errorCodeForMessage, localwhat, description, othertb)
        else:
            message = 'Code<{}>{}: {}'.format(errorCodeForMessage, localwhat, description)
        super().__init__(message)
        self.local = local
        self.errorCode = errorCode
        self.description = description
        self.fatal = fatal