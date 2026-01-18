import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
class FakeThreadedConcurrentUploader(ConcurrentUploader):

    def _start_upload_threads(self, results_queue, upload_id, worker_queue, filename):
        self.results_queue = results_queue
        self.worker_queue = worker_queue
        self.upload_id = upload_id

    def _wait_for_upload_threads(self, hash_chunks, result_queue, total_parts):
        for i in range(total_parts):
            hash_chunks[i] = b'foo'