import tempfile
from boto.compat import Queue
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.concurrent import ConcurrentUploader, ConcurrentDownloader
from boto.glacier.concurrent import UploadWorkerThread
from boto.glacier.concurrent import _END_SENTINEL
def _start_upload_threads(self, results_queue, upload_id, worker_queue, filename):
    self.results_queue = results_queue
    self.worker_queue = worker_queue
    self.upload_id = upload_id