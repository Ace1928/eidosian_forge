from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
@implementer(sip.ILocator)
class DummyLocator:

    def getAddress(self, logicalURL):
        return defer.succeed(sip.URL('server.com', port=5060))