import copy
import logging
import re
import threading
from s3transfer.bandwidth import BandwidthLimiter, LeakyBucket
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS, KB, MB
from s3transfer.copies import CopySubmissionTask
from s3transfer.delete import DeleteSubmissionTask
from s3transfer.download import DownloadSubmissionTask
from s3transfer.exceptions import CancelledError, FatalError
from s3transfer.futures import (
from s3transfer.upload import UploadSubmissionTask
from s3transfer.utils import (
def _add_operation_defaults(self, bucket, extra_args):
    add_s3express_defaults(bucket, extra_args)