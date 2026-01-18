import threading
from dulwich.client import TCPGitClient
from dulwich.repo import Repo
from ...tests import TestCase, TestCaseWithTransport
from ...transport import transport_server_registry
from ..server import BzrBackend, BzrTCPGitServer
class TestPresent(TestCase):

    def test_present(self):
        transport_server_registry.get('git')