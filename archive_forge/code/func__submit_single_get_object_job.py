import collections
import contextlib
import logging
import multiprocessing
import threading
import signal
from copy import deepcopy
import botocore.session
from botocore.config import Config
from s3transfer.constants import MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.constants import PROCESS_USER_AGENT
from s3transfer.compat import MAXINT
from s3transfer.compat import BaseManager
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import BaseTransferFuture
from s3transfer.futures import BaseTransferMeta
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import OSUtils
from s3transfer.utils import CallArgs
def _submit_single_get_object_job(self, download_file_request, temp_filename):
    self._notify_jobs_to_complete(download_file_request.transfer_id, 1)
    self._submit_get_object_job(transfer_id=download_file_request.transfer_id, bucket=download_file_request.bucket, key=download_file_request.key, temp_filename=temp_filename, offset=0, extra_args=download_file_request.extra_args, filename=download_file_request.filename)