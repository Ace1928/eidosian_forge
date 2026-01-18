import copy
import logging
import threading
from botocore.compat import six
from s3transfer.constants import KB, MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.utils import get_callbacks
from s3transfer.utils import signal_transferring
from s3transfer.utils import signal_not_transferring
from s3transfer.utils import CallArgs
from s3transfer.utils import OSUtils
from s3transfer.utils import TaskSemaphore
from s3transfer.utils import SlidingWindowSemaphore
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import FatalError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.futures import BoundedExecutor
from s3transfer.futures import TransferFuture
from s3transfer.futures import TransferMeta
from s3transfer.futures import TransferCoordinator
from s3transfer.download import DownloadSubmissionTask
from s3transfer.upload import UploadSubmissionTask
from s3transfer.copies import CopySubmissionTask
from s3transfer.delete import DeleteSubmissionTask
from s3transfer.bandwidth import LeakyBucket
from s3transfer.bandwidth import BandwidthLimiter
def _validate_attrs_are_nonzero(self):
    for attr, attr_val in self.__dict__.items():
        if attr_val is not None and attr_val <= 0:
            raise ValueError('Provided parameter %s of value %s must be greater than 0.' % (attr, attr_val))