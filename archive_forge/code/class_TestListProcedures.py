import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
class TestListProcedures(XMLRPC):
    """
    This is a resource which customizes procedure enumeration to be used by the
    tests of support for this customization.
    """

    def listProcedures(self):
        """
        Return a list of a single method this resource will claim to support.
        """
        return ['foo']