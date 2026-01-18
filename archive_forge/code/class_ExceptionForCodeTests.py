from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IResolver
from twisted.names.common import ResolverBase
from twisted.names.dns import EFORMAT, ENAME, ENOTIMP, EREFUSED, ESERVER, Query
from twisted.names.error import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
class ExceptionForCodeTests(SynchronousTestCase):
    """
    Tests for L{ResolverBase.exceptionForCode}.
    """

    def setUp(self):
        self.exceptionForCode = ResolverBase().exceptionForCode

    def test_eformat(self):
        """
        L{ResolverBase.exceptionForCode} converts L{EFORMAT} to
        L{DNSFormatError}.
        """
        self.assertIs(self.exceptionForCode(EFORMAT), DNSFormatError)

    def test_eserver(self):
        """
        L{ResolverBase.exceptionForCode} converts L{ESERVER} to
        L{DNSServerError}.
        """
        self.assertIs(self.exceptionForCode(ESERVER), DNSServerError)

    def test_ename(self):
        """
        L{ResolverBase.exceptionForCode} converts L{ENAME} to L{DNSNameError}.
        """
        self.assertIs(self.exceptionForCode(ENAME), DNSNameError)

    def test_enotimp(self):
        """
        L{ResolverBase.exceptionForCode} converts L{ENOTIMP} to
        L{DNSNotImplementedError}.
        """
        self.assertIs(self.exceptionForCode(ENOTIMP), DNSNotImplementedError)

    def test_erefused(self):
        """
        L{ResolverBase.exceptionForCode} converts L{EREFUSED} to
        L{DNSQueryRefusedError}.
        """
        self.assertIs(self.exceptionForCode(EREFUSED), DNSQueryRefusedError)

    def test_other(self):
        """
        L{ResolverBase.exceptionForCode} converts any other response code to
        L{DNSUnknownError}.
        """
        self.assertIs(self.exceptionForCode(object()), DNSUnknownError)