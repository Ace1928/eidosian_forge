from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
def _endpointTest(self, service):
    """
        Use L{Options} to parse a single service configuration parameter and
        verify that an endpoint of the correct type is added to the list for
        that service.
        """
    options = Options()
    options.parseOptions(['--' + service, 'tcp:1234'])
    self.assertEqual(len(options[service]), 1)
    self.assertIsInstance(options[service][0], endpoints.TCP4ServerEndpoint)