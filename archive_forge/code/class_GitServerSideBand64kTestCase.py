import os
import sys
import threading
from dulwich.tests import skipIf
from ...server import DictBackend, TCPGitServer
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase, require_git_version
@skipIf(sys.platform == 'win32', 'Broken on windows, with very long fail time.')
class GitServerSideBand64kTestCase(GitServerTestCase):
    """Tests for client/server compatibility with side-band-64k support."""
    min_git_version = (1, 7, 0, 2)

    def setUp(self):
        super().setUp()
        if os.name == 'nt':
            require_git_version((1, 9, 3))

    def _handlers(self):
        return None

    def _check_server(self, server):
        receive_pack_handler_cls = server.handlers[b'git-receive-pack']
        caps = receive_pack_handler_cls.capabilities()
        self.assertIn(b'side-band-64k', caps)