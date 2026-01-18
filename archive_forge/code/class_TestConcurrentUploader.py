import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
class TestConcurrentUploader(unittest.TestCase):

    def setUp(self):
        super(TestConcurrentUploader, self).setUp()
        self.stat_patch = mock.patch('os.stat')
        self.addCleanup(self.stat_patch.stop)
        self.stat_mock = self.stat_patch.start()
        self.stat_mock.return_value.st_size = 1024 * 1024 * 8

    def test_calculate_required_part_size(self):
        self.stat_mock.return_value.st_size = 1024 * 1024 * 8
        uploader = ConcurrentUploader(mock.Mock(), 'vault_name')
        total_parts, part_size = uploader._calculate_required_part_size(1024 * 1024 * 8)
        self.assertEqual(total_parts, 2)
        self.assertEqual(part_size, 4 * 1024 * 1024)

    def test_calculate_required_part_size_too_small(self):
        too_small = 1 * 1024 * 1024
        self.stat_mock.return_value.st_size = 1024 * 1024 * 1024
        uploader = ConcurrentUploader(mock.Mock(), 'vault_name', part_size=too_small)
        total_parts, part_size = uploader._calculate_required_part_size(1024 * 1024 * 1024)
        self.assertEqual(total_parts, 256)
        self.assertEqual(part_size, 4 * 1024 * 1024)

    def test_work_queue_is_correctly_populated(self):
        uploader = FakeThreadedConcurrentUploader(mock.MagicMock(), 'vault_name')
        uploader.upload('foofile')
        q = uploader.worker_queue
        items = [q.get() for i in range(q.qsize())]
        self.assertEqual(items[0], (0, 4 * 1024 * 1024))
        self.assertEqual(items[1], (1, 4 * 1024 * 1024))
        self.assertEqual(len(items), 12)

    def test_correct_low_level_api_calls(self):
        api_mock = mock.MagicMock()
        upload_id = '0898d645-ea45-4548-9a67-578f507ead49'
        initiate_upload_mock = mock.Mock(return_value={'UploadId': upload_id})
        api_mock.attach_mock(initiate_upload_mock, 'initiate_multipart_upload')
        uploader = FakeThreadedConcurrentUploader(api_mock, 'vault_name')
        uploader.upload('foofile')
        initiate_upload_mock.assert_called_with('vault_name', 4 * 1024 * 1024, None)
        api_mock.complete_multipart_upload.assert_called_with('vault_name', upload_id, mock.ANY, 8 * 1024 * 1024)

    def test_downloader_work_queue_is_correctly_populated(self):
        job = mock.MagicMock()
        job.archive_size = 8 * 1024 * 1024
        downloader = FakeThreadedConcurrentDownloader(job)
        downloader.download('foofile')
        q = downloader.worker_queue
        items = [q.get() for i in range(q.qsize())]
        self.assertEqual(items[0], (0, 4 * 1024 * 1024))
        self.assertEqual(items[1], (1, 4 * 1024 * 1024))
        self.assertEqual(len(items), 12)