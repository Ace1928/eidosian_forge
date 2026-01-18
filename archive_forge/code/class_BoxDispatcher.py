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
@implementer(IBoxReceiver)
class BoxDispatcher:
    """
    A L{BoxDispatcher} dispatches '_ask', '_answer', and '_error' L{AmpBox}es,
    both incoming and outgoing, to their appropriate destinations.

    Outgoing commands are converted into L{Deferred}s and outgoing boxes, and
    associated tracking state to fire those L{Deferred} when '_answer' boxes
    come back.  Incoming '_answer' and '_error' boxes are converted into
    callbacks and errbacks on those L{Deferred}s, respectively.

    Incoming '_ask' boxes are converted into method calls on a supplied method
    locator.

    @ivar _outstandingRequests: a dictionary mapping request IDs to
    L{Deferred}s which were returned for those requests.

    @ivar locator: an object with a L{CommandLocator.locateResponder} method
        that locates a responder function that takes a Box and returns a result
        (either a Box or a Deferred which fires one).

    @ivar boxSender: an object which can send boxes, via the L{_sendBoxCommand}
    method, such as an L{AMP} instance.
    @type boxSender: L{IBoxSender}
    """
    _failAllReason = None
    _outstandingRequests = None
    _counter = 0
    boxSender = None

    def __init__(self, locator):
        self._outstandingRequests = {}
        self.locator = locator

    def startReceivingBoxes(self, boxSender):
        """
        The given boxSender is going to start calling boxReceived on this
        L{BoxDispatcher}.

        @param boxSender: The L{IBoxSender} to send command responses to.
        """
        self.boxSender = boxSender

    def stopReceivingBoxes(self, reason):
        """
        No further boxes will be received here.  Terminate all currently
        outstanding command deferreds with the given reason.
        """
        self.failAllOutgoing(reason)

    def failAllOutgoing(self, reason):
        """
        Call the errback on all outstanding requests awaiting responses.

        @param reason: the Failure instance to pass to those errbacks.
        """
        self._failAllReason = reason
        OR = self._outstandingRequests.items()
        self._outstandingRequests = None
        for key, value in OR:
            value.errback(reason)

    def _nextTag(self):
        """
        Generate protocol-local serial numbers for _ask keys.

        @return: a string that has not yet been used on this connection.
        """
        self._counter += 1
        return b'%x' % (self._counter,)

    def _sendBoxCommand(self, command, box, requiresAnswer=True):
        """
        Send a command across the wire with the given C{amp.Box}.

        Mutate the given box to give it any additional keys (_command, _ask)
        required for the command and request/response machinery, then send it.

        If requiresAnswer is True, returns a C{Deferred} which fires when a
        response is received. The C{Deferred} is fired with an C{amp.Box} on
        success, or with an C{amp.RemoteAmpError} if an error is received.

        If the Deferred fails and the error is not handled by the caller of
        this method, the failure will be logged and the connection dropped.

        @param command: a C{bytes}, the name of the command to issue.

        @param box: an AmpBox with the arguments for the command.

        @param requiresAnswer: a boolean.  Defaults to True.  If True, return a
        Deferred which will fire when the other side responds to this command.
        If False, return None and do not ask the other side for acknowledgement.

        @return: a Deferred which fires the AmpBox that holds the response to
        this command, or None, as specified by requiresAnswer.

        @raise ProtocolSwitched: if the protocol has been switched.
        """
        if self._failAllReason is not None:
            if requiresAnswer:
                return fail(self._failAllReason)
            else:
                return None
        box[COMMAND] = command
        tag = self._nextTag()
        if requiresAnswer:
            box[ASK] = tag
        box._sendTo(self.boxSender)
        if requiresAnswer:
            result = self._outstandingRequests[tag] = Deferred()
        else:
            result = None
        return result

    def callRemoteString(self, command, requiresAnswer=True, **kw):
        """
        This is a low-level API, designed only for optimizing simple messages
        for which the overhead of parsing is too great.

        @param command: a C{bytes} naming the command.

        @param kw: arguments to the amp box.

        @param requiresAnswer: a boolean.  Defaults to True.  If True, return a
        Deferred which will fire when the other side responds to this command.
        If False, return None and do not ask the other side for acknowledgement.

        @return: a Deferred which fires the AmpBox that holds the response to
        this command, or None, as specified by requiresAnswer.
        """
        box = Box(kw)
        return self._sendBoxCommand(command, box, requiresAnswer)

    def callRemote(self, commandType, *a, **kw):
        """
        This is the primary high-level API for sending messages via AMP.  Invoke it
        with a command and appropriate arguments to send a message to this
        connection's peer.

        @param commandType: a subclass of Command.
        @type commandType: L{type}

        @param a: Positional (special) parameters taken by the command.
        Positional parameters will typically not be sent over the wire.  The
        only command included with AMP which uses positional parameters is
        L{ProtocolSwitchCommand}, which takes the protocol that will be
        switched to as its first argument.

        @param kw: Keyword arguments taken by the command.  These are the
        arguments declared in the command's 'arguments' attribute.  They will
        be encoded and sent to the peer as arguments for the L{commandType}.

        @return: If L{commandType} has a C{requiresAnswer} attribute set to
        L{False}, then return L{None}.  Otherwise, return a L{Deferred} which
        fires with a dictionary of objects representing the result of this
        call.  Additionally, this L{Deferred} may fail with an exception
        representing a connection failure, with L{UnknownRemoteError} if the
        other end of the connection fails for an unknown reason, or with any
        error specified as a key in L{commandType}'s C{errors} dictionary.
        """
        try:
            co = commandType(*a, **kw)
        except BaseException:
            return fail()
        return co._doCommand(self)

    def unhandledError(self, failure):
        """
        This is a terminal callback called after application code has had a
        chance to quash any errors.
        """
        return self.boxSender.unhandledError(failure)

    def _answerReceived(self, box):
        """
        An AMP box was received that answered a command previously sent with
        L{callRemote}.

        @param box: an AmpBox with a value for its L{ANSWER} key.
        """
        question = self._outstandingRequests.pop(box[ANSWER])
        question.addErrback(self.unhandledError)
        question.callback(box)

    def _errorReceived(self, box):
        """
        An AMP box was received that answered a command previously sent with
        L{callRemote}, with an error.

        @param box: an L{AmpBox} with a value for its L{ERROR}, L{ERROR_CODE},
        and L{ERROR_DESCRIPTION} keys.
        """
        question = self._outstandingRequests.pop(box[ERROR])
        question.addErrback(self.unhandledError)
        errorCode = box[ERROR_CODE]
        description = box[ERROR_DESCRIPTION]
        if isinstance(description, bytes):
            description = description.decode('utf-8', 'replace')
        if errorCode in PROTOCOL_ERRORS:
            exc = PROTOCOL_ERRORS[errorCode](errorCode, description)
        else:
            exc = RemoteAmpError(errorCode, description)
        question.errback(Failure(exc))

    def _commandReceived(self, box):
        """
        @param box: an L{AmpBox} with a value for its L{COMMAND} and L{ASK}
        keys.
        """

        def formatAnswer(answerBox):
            answerBox[ANSWER] = box[ASK]
            return answerBox

        def formatError(error):
            if error.check(RemoteAmpError):
                code = error.value.errorCode
                desc = error.value.description
                if isinstance(desc, str):
                    desc = desc.encode('utf-8', 'replace')
                if error.value.fatal:
                    errorBox = QuitBox()
                else:
                    errorBox = AmpBox()
            else:
                errorBox = QuitBox()
                log.err(error)
                code = UNKNOWN_ERROR_CODE
                desc = b'Unknown Error'
            errorBox[ERROR] = box[ASK]
            errorBox[ERROR_DESCRIPTION] = desc
            errorBox[ERROR_CODE] = code
            return errorBox
        deferred = self.dispatchCommand(box)
        if ASK in box:
            deferred.addCallbacks(formatAnswer, formatError)
            deferred.addCallback(self._safeEmit)
        deferred.addErrback(self.unhandledError)

    def ampBoxReceived(self, box):
        """
        An AmpBox was received, representing a command, or an answer to a
        previously issued command (either successful or erroneous).  Respond to
        it according to its contents.

        @param box: an AmpBox

        @raise NoEmptyBoxes: when a box is received that does not contain an
        '_answer', '_command' / '_ask', or '_error' key; i.e. one which does not
        fit into the command / response protocol defined by AMP.
        """
        if ANSWER in box:
            self._answerReceived(box)
        elif ERROR in box:
            self._errorReceived(box)
        elif COMMAND in box:
            self._commandReceived(box)
        else:
            raise NoEmptyBoxes(box)

    def _safeEmit(self, aBox):
        """
        Emit a box, ignoring L{ProtocolSwitched} and L{ConnectionLost} errors
        which cannot be usefully handled.
        """
        try:
            aBox._sendTo(self.boxSender)
        except (ProtocolSwitched, ConnectionLost):
            pass

    def dispatchCommand(self, box):
        """
        A box with a _command key was received.

        Dispatch it to a local handler call it.

        @param box: an AmpBox to be dispatched.
        """
        cmd = box[COMMAND]
        responder = self.locator.locateResponder(cmd)
        if responder is None:
            description = f'Unhandled Command: {cmd!r}'
            return fail(RemoteAmpError(UNHANDLED_ERROR_CODE, description, False, local=Failure(UnhandledCommand())))
        return maybeDeferred(responder, box)