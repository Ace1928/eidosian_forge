import contextlib
import threading
from kazoo.protocol import paths as k_paths
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from zake import utils as zake_utils
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_zookeeper
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.types import entity
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
@testtools.skipIf(not ZOOKEEPER_AVAILABLE, 'zookeeper is not available')
class ZookeeperJobboardTest(test.TestCase, ZookeeperBoardTestMixin):

    def create_board(self, persistence=None):

        def cleanup_path(client, path):
            if not client.connected:
                return
            client.delete(path, recursive=True)
        client = kazoo_utils.make_client(test_utils.ZK_TEST_CONFIG.copy())
        path = TEST_PATH_TPL % uuidutils.generate_uuid()
        board = impl_zookeeper.ZookeeperJobBoard('test-board', {'path': path}, client=client, persistence=persistence)
        self.addCleanup(self.close_client, client)
        self.addCleanup(cleanup_path, client, path)
        self.addCleanup(board.close)
        return (client, board)

    def setUp(self):
        super(ZookeeperJobboardTest, self).setUp()
        self.client, self.board = self.create_board()