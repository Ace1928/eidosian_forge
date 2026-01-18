from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
class MakeServiceTests(TestCase):
    """
    Tests for L{twisted.mail.tap.makeService}
    """

    def _endpointServerTest(self, key, factoryClass):
        """
        Configure a service with two endpoints for the protocol associated with
        C{key} and verify that when the service is started a factory of type
        C{factoryClass} is used to listen on each of them.
        """
        cleartext = SpyEndpoint()
        secure = SpyEndpoint()
        config = Options()
        config[key] = [cleartext, secure]
        service = makeService(config)
        service.privilegedStartService()
        service.startService()
        self.addCleanup(service.stopService)
        self.assertIsInstance(cleartext.listeningWith, factoryClass)
        self.assertIsInstance(secure.listeningWith, factoryClass)

    def test_pop3(self):
        """
        If one or more endpoints is included in the configuration passed to
        L{makeService} for the C{"pop3"} key, a service for starting a POP3
        server is constructed for each of them and attached to the returned
        service.
        """
        self._endpointServerTest('pop3', protocols.POP3Factory)

    def test_smtp(self):
        """
        If one or more endpoints is included in the configuration passed to
        L{makeService} for the C{"smtp"} key, a service for starting an SMTP
        server is constructed for each of them and attached to the returned
        service.
        """
        self._endpointServerTest('smtp', protocols.SMTPFactory)