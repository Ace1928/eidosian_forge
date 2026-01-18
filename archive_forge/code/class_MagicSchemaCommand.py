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
class MagicSchemaCommand(amp.Command):
    """
    A command which overrides L{parseResponse}, L{parseArguments}, and
    L{makeResponse}.
    """

    @classmethod
    def parseResponse(self, strings, protocol):
        """
        Don't do any parsing, just jam the input strings and protocol
        onto the C{protocol.parseResponseArguments} attribute as a
        two-tuple. Return the original strings.
        """
        protocol.parseResponseArguments = (strings, protocol)
        return strings

    @classmethod
    def parseArguments(cls, strings, protocol):
        """
        Don't do any parsing, just jam the input strings and protocol
        onto the C{protocol.parseArgumentsArguments} attribute as a
        two-tuple. Return the original strings.
        """
        protocol.parseArgumentsArguments = (strings, protocol)
        return strings

    @classmethod
    def makeArguments(cls, objects, protocol):
        """
        Don't do any serializing, just jam the input strings and protocol
        onto the C{protocol.makeArgumentsArguments} attribute as a
        two-tuple. Return the original strings.
        """
        protocol.makeArgumentsArguments = (objects, protocol)
        return objects