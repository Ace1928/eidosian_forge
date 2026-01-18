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
def _start_get_object_workers(self):
    logger.debug('Starting %s GetObjectWorkers.', self._transfer_config.max_request_processes)
    for _ in range(self._transfer_config.max_request_processes):
        worker = GetObjectWorker(queue=self._worker_queue, client_factory=self._client_factory, transfer_monitor=self._transfer_monitor, osutil=self._osutil)
        worker.start()
        self._workers.append(worker)