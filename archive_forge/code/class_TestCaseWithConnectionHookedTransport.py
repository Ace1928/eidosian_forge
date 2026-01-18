from ..transport import Transport
from . import test_sftp_transport
class TestCaseWithConnectionHookedTransport(test_sftp_transport.TestCaseWithSFTPServer):

    def setUp(self):
        super().setUp()
        self.reset_connections()

    def start_logging_connections(self):
        Transport.hooks.install_named_hook('post_connect', self.connections.append, None)

    def reset_connections(self):
        self.connections = []