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
class ProcessTransferConfig(object):

    def __init__(self, multipart_threshold=8 * MB, multipart_chunksize=8 * MB, max_request_processes=10):
        """Configuration for the ProcessPoolDownloader

        :param multipart_threshold: The threshold for which ranged downloads
            occur.

        :param multipart_chunksize: The chunk size of each ranged download.

        :param max_request_processes: The maximum number of processes that
            will be making S3 API transfer-related requests at a time.
        """
        self.multipart_threshold = multipart_threshold
        self.multipart_chunksize = multipart_chunksize
        self.max_request_processes = max_request_processes