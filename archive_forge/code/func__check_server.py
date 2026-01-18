import os
import sys
import threading
from dulwich.tests import skipIf
from ...server import DictBackend, TCPGitServer
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase, require_git_version
def _check_server(self, server):
    receive_pack_handler_cls = server.handlers[b'git-receive-pack']
    caps = receive_pack_handler_cls.capabilities()
    self.assertIn(b'side-band-64k', caps)