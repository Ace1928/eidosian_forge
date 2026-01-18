import logging
import threading
import heapq
from botocore.compat import six
from s3transfer.compat import seekable
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import get_callbacks
from s3transfer.utils import invoke_progress_callbacks
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import FunctionContainer
from s3transfer.utils import CountCallbackInvoker
from s3transfer.utils import StreamReaderProgress
from s3transfer.utils import DeferredOpenFile
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
class IOStreamingWriteTask(Task):
    """Task for writing data to a non-seekable stream."""

    def _main(self, fileobj, data):
        """Write data to a fileobj.

        Data will be written directly to the fileboj without
        any prior seeking.

        :param fileobj: The fileobj to write content to
        :param data: The data to write

        """
        fileobj.write(data)