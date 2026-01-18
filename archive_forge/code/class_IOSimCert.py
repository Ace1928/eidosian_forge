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
class IOSimCert:
    verifyCount = 0

    def options(self, *ign):
        return self

    def iosimVerify(self, otherCert):
        """
        This isn't a real certificate, and wouldn't work on a real socket, but
        iosim specifies a different API so that we don't have to do any crypto
        math to demonstrate that the right functions get called in the right
        places.
        """
        assert otherCert is self
        self.verifyCount += 1
        return True