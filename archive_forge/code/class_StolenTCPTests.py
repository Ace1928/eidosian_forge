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
class StolenTCPTests(ProperlyCloseFilesMixin, TestCase):
    """
    For SSL transports, test many of the same things which are tested for
    TCP transports.
    """
    if interfaces.IReactorSSL(reactor, None) is None:
        skip = 'Reactor does not support SSL, cannot run SSL tests'

    def createServer(self, address, portNumber, factory):
        """
        Create an SSL server with a certificate using L{IReactorSSL.listenSSL}.
        """
        cert = ssl.PrivateCertificate.loadPEM(FilePath(certPath).getContent())
        contextFactory = cert.options()
        return reactor.listenSSL(portNumber, factory, contextFactory, interface=address)

    def connectClient(self, address, portNumber, clientCreator):
        """
        Create an SSL client using L{IReactorSSL.connectSSL}.
        """
        contextFactory = ssl.CertificateOptions()
        return clientCreator.connectSSL(address, portNumber, contextFactory)

    def getHandleExceptionType(self):
        """
        Return L{OpenSSL.SSL.Error} as the expected error type which will be
        raised by a write to the L{OpenSSL.SSL.Connection} object after it has
        been closed.
        """
        return SSL.Error

    def getHandleErrorCodeMatcher(self):
        """
        Return a L{hamcrest.core.matcher.Matcher} for the argument
        L{OpenSSL.SSL.Error} will be constructed with for this case.
        This is basically just a random OpenSSL implementation detail.
        It would be better if this test worked in a way which did not
        require this.
        """
        return hamcrest.contains(hamcrest.contains(hamcrest.equal_to('SSL routines'), hamcrest.any_of(hamcrest.equal_to('SSL_write'), hamcrest.equal_to('ssl_write_internal'), hamcrest.equal_to('')), hamcrest.equal_to('protocol is shutdown')))