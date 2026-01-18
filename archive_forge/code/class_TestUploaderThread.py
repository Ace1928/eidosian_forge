import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
class TestUploaderThread(unittest.TestCase):

    def setUp(self):
        self.fileobj = tempfile.NamedTemporaryFile()
        self.filename = self.fileobj.name

    def test_fileobj_closed_when_thread_shuts_down(self):
        thread = UploadWorkerThread(mock.Mock(), 'vault_name', self.filename, 'upload_id', Queue(), Queue())
        fileobj = thread._fileobj
        self.assertFalse(fileobj.closed)
        thread.should_continue = False
        thread.run()
        self.assertTrue(fileobj.closed)

    def test_upload_errors_have_exception_messages(self):
        api = mock.Mock()
        job_queue = Queue()
        result_queue = Queue()
        upload_thread = UploadWorkerThread(api, 'vault_name', self.filename, 'upload_id', job_queue, result_queue, num_retries=1, time_between_retries=0)
        api.upload_part.side_effect = Exception('exception message')
        job_queue.put((0, 1024))
        job_queue.put(_END_SENTINEL)
        upload_thread.run()
        result = result_queue.get(timeout=1)
        self.assertIn('exception message', str(result))

    def test_num_retries_is_obeyed(self):
        api = mock.Mock()
        job_queue = Queue()
        result_queue = Queue()
        upload_thread = UploadWorkerThread(api, 'vault_name', self.filename, 'upload_id', job_queue, result_queue, num_retries=2, time_between_retries=0)
        api.upload_part.side_effect = Exception()
        job_queue.put((0, 1024))
        job_queue.put(_END_SENTINEL)
        upload_thread.run()
        self.assertEqual(api.upload_part.call_count, 3)