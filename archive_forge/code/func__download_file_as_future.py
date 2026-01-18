import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
def _download_file_as_future(self, bucket, key, filename, object_size, callback):
    part_size = self._config.multipart_chunksize
    num_parts = int(math.ceil(object_size / float(part_size)))
    max_workers = self._config.max_concurrency
    download_partial = functools.partial(self._download_range, bucket, key, filename, part_size, num_parts, callback)
    try:
        with self._executor_cls(max_workers=max_workers) as executor:
            list(executor.map(download_partial, range(num_parts)))
    finally:
        self._ioqueue.put(SHUTDOWN_SENTINEL)