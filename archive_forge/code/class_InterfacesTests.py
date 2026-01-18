from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class InterfacesTests(unittest.TestCase):

    def test_interface(self):
        """
        L{telnet.TelnetProtocol} implements L{telnet.ITelnetProtocol}
        """
        p = telnet.TelnetProtocol()
        verifyObject(telnet.ITelnetProtocol, p)