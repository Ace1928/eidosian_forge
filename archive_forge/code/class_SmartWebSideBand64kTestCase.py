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
class SmartWebSideBand64kTestCase(SmartWebTestCase):
    """Test cases for smart HTTP server with side-band-64k support."""
    min_git_version = (1, 7, 0, 2)

    def setUp(self):
        self.o_uph_cap = patch_capabilities(UploadPackHandler, (b'no-done',))
        self.o_rph_cap = patch_capabilities(ReceivePackHandler, (b'no-done',))
        super().setUp()

    def tearDown(self):
        super().tearDown()
        UploadPackHandler.capabilities = self.o_uph_cap
        ReceivePackHandler.capabilities = self.o_rph_cap

    def _handlers(self):
        return None

    def _check_app(self, app):
        receive_pack_handler_cls = app.handlers[b'git-receive-pack']
        caps = receive_pack_handler_cls.capabilities()
        self.assertIn(b'side-band-64k', caps)
        self.assertNotIn(b'no-done', caps)