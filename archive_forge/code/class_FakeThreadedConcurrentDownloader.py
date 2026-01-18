import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
class FakeThreadedConcurrentDownloader(ConcurrentDownloader):

    def _start_download_threads(self, results_queue, worker_queue):
        self.results_queue = results_queue
        self.worker_queue = worker_queue

    def _wait_for_download_threads(self, filename, result_queue, total_parts):
        pass