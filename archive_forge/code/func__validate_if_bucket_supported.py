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
def _validate_if_bucket_supported(self, bucket):
    if self.VALIDATE_SUPPORTED_BUCKET_VALUES:
        for resource, pattern in self._UNSUPPORTED_BUCKET_PATTERNS.items():
            match = pattern.match(bucket)
            if match:
                raise ValueError('TransferManager methods do not support %s resource. Use direct client calls instead.' % resource)