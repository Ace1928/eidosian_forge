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
class ZookeeperBoardTestMixin(base.BoardTestMixin):

    def close_client(self, client):
        kazoo_utils.finalize_client(client)

    @contextlib.contextmanager
    def flush(self, client, path=None):
        if not path:
            path = FLUSH_PATH_TPL % uuidutils.generate_uuid()
        created = threading.Event()
        deleted = threading.Event()

        def on_created(data, stat):
            if stat is not None:
                created.set()
                return False

        def on_deleted(data, stat):
            if stat is None:
                deleted.set()
                return False
        watchers.DataWatch(client, path, func=on_created)
        client.create(path, makepath=True)
        if not created.wait(test_utils.WAIT_TIMEOUT):
            raise RuntimeError('Could not receive creation of %s in the alloted timeout of %s seconds' % (path, test_utils.WAIT_TIMEOUT))
        try:
            yield
        finally:
            watchers.DataWatch(client, path, func=on_deleted)
            client.delete(path, recursive=True)
            if not deleted.wait(test_utils.WAIT_TIMEOUT):
                raise RuntimeError('Could not receive deletion of %s in the alloted timeout of %s seconds' % (path, test_utils.WAIT_TIMEOUT))

    def test_posting_no_post(self):
        with base.connect_close(self.board):
            with mock.patch.object(self.client, 'create') as create_func:
                create_func.side_effect = IOError('Unable to post')
                self.assertRaises(IOError, self.board.post, 'test', p_utils.temporary_log_book())
            self.assertEqual(0, self.board.job_count)

    def test_board_iter(self):
        with base.connect_close(self.board):
            it = self.board.iterjobs()
            self.assertEqual(self.board, it.board)
            self.assertFalse(it.only_unclaimed)
            self.assertFalse(it.ensure_fresh)

    @mock.patch('taskflow.jobs.backends.impl_zookeeper.misc.millis_to_datetime')
    def test_posting_dates(self, mock_dt):
        epoch = misc.millis_to_datetime(0)
        mock_dt.return_value = epoch
        with base.connect_close(self.board):
            j = self.board.post('test', p_utils.temporary_log_book())
            self.assertEqual(epoch, j.created_on)
            self.assertEqual(epoch, j.last_modified)
        self.assertTrue(mock_dt.called)