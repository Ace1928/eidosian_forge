import math
from botocore.compat import six
from s3transfer.compat import seekable, readable
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import DeferredOpenFile, ChunksizeAdjuster
class AggregatedProgressCallback(object):

    def __init__(self, callbacks, threshold=1024 * 256):
        """Aggregates progress updates for every provided progress callback

        :type callbacks: A list of functions that accepts bytes_transferred
            as a single argument
        :param callbacks: The callbacks to invoke when threshold is reached

        :type threshold: int
        :param threshold: The progress threshold in which to take the
            aggregated progress and invoke the progress callback with that
            aggregated progress total
        """
        self._callbacks = callbacks
        self._threshold = threshold
        self._bytes_seen = 0

    def __call__(self, bytes_transferred):
        self._bytes_seen += bytes_transferred
        if self._bytes_seen >= self._threshold:
            self._trigger_callbacks()

    def flush(self):
        """Flushes out any progress that has not been sent to its callbacks"""
        if self._bytes_seen > 0:
            self._trigger_callbacks()

    def _trigger_callbacks(self):
        for callback in self._callbacks:
            callback(bytes_transferred=self._bytes_seen)
        self._bytes_seen = 0