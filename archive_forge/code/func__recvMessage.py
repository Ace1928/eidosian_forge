import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
def _recvMessage(self, findObjMethod, requestID, objectID, message, answerRequired, netArgs, netKw):
    """
        Received a message-send.

        Look up message based on object, unserialize the arguments, and
        invoke it with args, and send an 'answer' or 'error' response.

        @param findObjMethod: A callable which takes C{objectID} as argument.
        @param requestID: The requiest ID.
        @param objectID: The object ID.
        @param message: The message.
        @param answerRequired:
        @param netArgs: Arguments.
        @param netKw: Keyword arguments.
        """
    if not isinstance(message, str):
        message = message.decode('utf8')
    try:
        object = findObjMethod(objectID)
        if object is None:
            raise Error('Invalid Object ID')
        netResult = object.remoteMessageReceived(self, message, netArgs, netKw)
    except Error as e:
        if answerRequired:
            if isinstance(e, Jellyable) or self.security.isClassAllowed(e.__class__):
                self._sendError(e, requestID)
            else:
                self._sendError(CopyableFailure(e), requestID)
    except BaseException:
        if answerRequired:
            log.msg('Peer will receive following PB traceback:', isError=True)
            f = CopyableFailure()
            self._sendError(f, requestID)
        log.err()
    else:
        if answerRequired:
            if isinstance(netResult, defer.Deferred):
                args = (requestID,)
                netResult.addCallbacks(self._sendAnswer, self._sendFailureOrError, callbackArgs=args, errbackArgs=args)
            else:
                self._sendAnswer(netResult, requestID)