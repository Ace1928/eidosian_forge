from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
class BypassTLSTests(unittest.TestCase):
    """
    Tests for the L{_newtls._BypassTLS} class.
    """
    if not _newtls:
        skip = "Couldn't import _newtls, perhaps pyOpenSSL is old or missing"

    def test_loseConnectionPassThrough(self):
        """
        C{_BypassTLS.loseConnection} calls C{loseConnection} on the base
        class, while preserving any default argument in the base class'
        C{loseConnection} implementation.
        """
        default = object()
        result = []

        class FakeTransport:

            def loseConnection(self, _connDone=default):
                result.append(_connDone)
        bypass = _newtls._BypassTLS(FakeTransport, FakeTransport())
        bypass.loseConnection()
        self.assertEqual(result, [default])
        notDefault = object()
        bypass.loseConnection(notDefault)
        self.assertEqual(result, [default, notDefault])