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
class DumbWebTestCase(WebTests, CompatTestCase):
    """Test cases for dumb HTTP server."""

    def _make_app(self, backend):
        return make_wsgi_chain(backend, dumb=True)

    def test_push_to_dulwich(self):
        raise SkipTest('Dumb web pushing not supported.')

    def test_push_to_dulwich_remove_branch(self):
        raise SkipTest('Dumb web pushing not supported.')

    def test_new_shallow_clone_from_dulwich(self):
        raise SkipTest('Dumb web shallow cloning not supported.')

    def test_shallow_clone_from_git_is_identical(self):
        raise SkipTest('Dumb web shallow cloning not supported.')

    def test_fetch_same_depth_into_shallow_clone_from_dulwich(self):
        raise SkipTest('Dumb web shallow cloning not supported.')

    def test_fetch_full_depth_into_shallow_clone_from_dulwich(self):
        raise SkipTest('Dumb web shallow cloning not supported.')

    def test_push_to_dulwich_issue_88_standard(self):
        raise SkipTest('Dumb web pushing not supported.')