from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
@implementer(sip.ILocator)
class FailingLocator:

    def getAddress(self, logicalURL):
        return defer.fail(LookupError())