import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class IPv6AddressTests(SynchronousTestCase, AddressTestCaseMixin):
    addressArgSpec = (('type', '%s'), ('host', '%r'), ('port', '%d'))

    def buildAddress(self):
        """
        Create an arbitrary new L{IPv6Address} instance with a C{"TCP"}
        type.  A new instance is created for each call, but always for the
        same address.
        """
        return IPv6Address('TCP', '::1', 0)

    def buildDifferentAddress(self):
        """
        Like L{buildAddress}, but with a different fixed address.
        """
        return IPv6Address('TCP', '::2', 0)