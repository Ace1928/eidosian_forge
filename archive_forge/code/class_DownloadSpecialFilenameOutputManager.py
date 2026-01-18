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
class DownloadSpecialFilenameOutputManager(DownloadNonSeekableOutputManager):

    def __init__(self, osutil, transfer_coordinator, io_executor, defer_queue=None):
        super(DownloadSpecialFilenameOutputManager, self).__init__(osutil, transfer_coordinator, io_executor, defer_queue)
        self._fileobj = None

    @classmethod
    def is_compatible(cls, download_target, osutil):
        return isinstance(download_target, six.string_types) and osutil.is_special_file(download_target)

    def get_fileobj_for_io_writes(self, transfer_future):
        filename = transfer_future.meta.call_args.fileobj
        self._fileobj = self._get_fileobj_from_filename(filename)
        return self._fileobj

    def get_final_io_task(self):
        return IOCloseTask(transfer_coordinator=self._transfer_coordinator, is_final=True, main_kwargs={'fileobj': self._fileobj})