import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
class SpammyTLSTests(TLSTests):
    """
    Test TLS features with bytes sitting in the out buffer.
    """
    if interfaces.IReactorSSL(reactor, None) is None:
        skip = 'Reactor does not support SSL, cannot run SSL tests'
    fillBuffer = True