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
class DownloadFilenameOutputManager(DownloadOutputManager):

    def __init__(self, osutil, transfer_coordinator, io_executor):
        super(DownloadFilenameOutputManager, self).__init__(osutil, transfer_coordinator, io_executor)
        self._final_filename = None
        self._temp_filename = None
        self._temp_fileobj = None

    @classmethod
    def is_compatible(cls, download_target, osutil):
        return isinstance(download_target, six.string_types)

    def get_fileobj_for_io_writes(self, transfer_future):
        fileobj = transfer_future.meta.call_args.fileobj
        self._final_filename = fileobj
        self._temp_filename = self._osutil.get_temp_filename(fileobj)
        self._temp_fileobj = self._get_temp_fileobj()
        return self._temp_fileobj

    def get_final_io_task(self):
        return IORenameFileTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'fileobj': self._temp_fileobj, 'final_filename': self._final_filename, 'osutil': self._osutil}, is_final=True)

    def _get_temp_fileobj(self):
        f = self._get_fileobj_from_filename(self._temp_filename)
        self._transfer_coordinator.add_failure_cleanup(self._osutil.remove_file, self._temp_filename)
        return f