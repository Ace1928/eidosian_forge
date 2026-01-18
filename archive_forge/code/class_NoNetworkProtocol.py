import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class NoNetworkProtocol(amp.AMP):
    """
    An L{amp.AMP} subclass which overrides private methods to avoid
    testing the network. It also provides a responder for
    L{MagicSchemaCommand} that does nothing, so that tests can test
    aspects of the interaction of L{amp.Command}s and L{amp.AMP}.

    @ivar parseArgumentsArguments: Arguments that have been passed to any
    L{MagicSchemaCommand}, if L{MagicSchemaCommand} has been handled by
    this protocol.

    @ivar parseResponseArguments: Responses that have been returned from a
    L{MagicSchemaCommand}, if L{MagicSchemaCommand} has been handled by
    this protocol.

    @ivar makeArgumentsArguments: Arguments that have been serialized by any
    L{MagicSchemaCommand}, if L{MagicSchemaCommand} has been handled by
    this protocol.
    """

    def _sendBoxCommand(self, commandName, strings, requiresAnswer):
        """
        Return a Deferred which fires with the original strings.
        """
        return defer.succeed(strings)
    MagicSchemaCommand.responder(lambda s, weird: {})