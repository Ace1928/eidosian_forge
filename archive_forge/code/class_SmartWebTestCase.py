import sys
import threading
from typing import Tuple
from wsgiref import simple_server
from dulwich.tests import SkipTest, skipIf
from ...server import DictBackend, ReceivePackHandler, UploadPackHandler
from ...web import (
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase
@skipIf(sys.platform == 'win32', 'Broken on windows, with very long fail time.')
class SmartWebTestCase(WebTests, CompatTestCase):
    """Test cases for smart HTTP server.

    This server test case does not use side-band-64k in git-receive-pack.
    """
    min_git_version: Tuple[int, ...] = (1, 6, 6)

    def _handlers(self):
        return {b'git-receive-pack': NoSideBand64kReceivePackHandler}

    def _check_app(self, app):
        receive_pack_handler_cls = app.handlers[b'git-receive-pack']
        caps = receive_pack_handler_cls.capabilities()
        self.assertNotIn(b'side-band-64k', caps)

    def _make_app(self, backend):
        app = make_wsgi_chain(backend, handlers=self._handlers())
        to_check = app
        while not issubclass(to_check.__class__, HTTPGitApplication):
            to_check = to_check.app
        self._check_app(to_check)
        return app