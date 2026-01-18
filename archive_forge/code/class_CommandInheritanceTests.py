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
class CommandInheritanceTests(TestCase):
    """
    These tests verify that commands inherit error conditions properly.
    """

    def errorCheck(self, err, proto, cmd, **kw):
        """
        Check that the appropriate kind of error is raised when a given command
        is sent to a given protocol.
        """
        c, s, p = connectedServerAndClient(ServerClass=proto, ClientClass=proto)
        d = c.callRemote(cmd, **kw)
        d2 = self.failUnlessFailure(d, err)
        p.flush()
        return d2

    def test_basicErrorPropagation(self):
        """
        Verify that errors specified in a superclass are respected normally
        even if it has subclasses.
        """
        return self.errorCheck(InheritedError, NormalCommandProtocol, BaseCommand)

    def test_inheritedErrorPropagation(self):
        """
        Verify that errors specified in a superclass command are propagated to
        its subclasses.
        """
        return self.errorCheck(InheritedError, InheritedCommandProtocol, InheritedCommand)

    def test_inheritedErrorAddition(self):
        """
        Verify that new errors specified in a subclass of an existing command
        are honored even if the superclass defines some errors.
        """
        return self.errorCheck(OtherInheritedError, AddedCommandProtocol, AddErrorsCommand, other=True)

    def test_additionWithOriginalError(self):
        """
        Verify that errors specified in a command's superclass are respected
        even if that command defines new errors itself.
        """
        return self.errorCheck(InheritedError, AddedCommandProtocol, AddErrorsCommand, other=False)