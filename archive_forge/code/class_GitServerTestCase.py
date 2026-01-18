import os
import sys
import threading
from dulwich.tests import skipIf
from ...server import DictBackend, TCPGitServer
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase, require_git_version
@skipIf(sys.platform == 'win32', 'Broken on windows, with very long fail time.')
class GitServerTestCase(ServerTests, CompatTestCase):
    """Tests for client/server compatibility.

    This server test case does not use side-band-64k in git-receive-pack.
    """
    protocol = 'git'

    def _handlers(self):
        return {b'git-receive-pack': NoSideBand64kReceivePackHandler}

    def _check_server(self, dul_server):
        receive_pack_handler_cls = dul_server.handlers[b'git-receive-pack']
        caps = receive_pack_handler_cls.capabilities()
        self.assertNotIn(b'side-band-64k', caps)

    def _start_server(self, repo):
        backend = DictBackend({b'/': repo})
        dul_server = TCPGitServer(backend, b'localhost', 0, handlers=self._handlers())
        self._check_server(dul_server)
        self.addCleanup(dul_server.shutdown)
        self.addCleanup(dul_server.server_close)
        threading.Thread(target=dul_server.serve).start()
        self._server = dul_server
        _, port = self._server.socket.getsockname()
        return port