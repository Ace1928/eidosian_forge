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
def _wrapWithSerialization(self, aCallable, command):
    """
        Wrap aCallable with its command's argument de-serialization
        and result serialization logic.

        @param aCallable: a callable with a 'command' attribute, designed to be
        called with keyword arguments.

        @param command: the command class whose serialization to use.

        @return: a 1-arg callable which, when invoked with an AmpBox, will
        deserialize the argument list and invoke appropriate user code for the
        callable's command, returning a Deferred which fires with the result or
        fails with an error.
        """

    def doit(box):
        kw = command.parseArguments(box, self)

        def checkKnownErrors(error):
            key = error.trap(*command.allErrors)
            code = command.allErrors[key]
            desc = str(error.value)
            return Failure(RemoteAmpError(code, desc, key in command.fatalErrors, local=error))

        def makeResponseFor(objects):
            try:
                return command.makeResponse(objects, self)
            except BaseException:
                originalFailure = Failure()
                raise BadLocalReturn('%r returned %r and %r could not serialize it' % (aCallable, objects, command), originalFailure)
        return maybeDeferred(aCallable, **kw).addCallback(makeResponseFor).addErrback(checkKnownErrors)
    return doit