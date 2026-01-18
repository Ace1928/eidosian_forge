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
def _run_get_object_job(self, job):
    try:
        self._do_get_object(bucket=job.bucket, key=job.key, temp_filename=job.temp_filename, extra_args=job.extra_args, offset=job.offset)
    except Exception as e:
        logger.debug('Exception caught when downloading object for get object job %s: %s', job, e, exc_info=True)
        self._transfer_monitor.notify_exception(job.transfer_id, e)