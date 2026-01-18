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
def _upload_parts(self, upload_id, filename, bucket, key, callback, extra_args):
    upload_parts_extra_args = self._extra_upload_part_args(extra_args)
    parts = []
    part_size = self._config.multipart_chunksize
    num_parts = int(math.ceil(self._os.get_file_size(filename) / float(part_size)))
    max_workers = self._config.max_concurrency
    with self._executor_cls(max_workers=max_workers) as executor:
        upload_partial = functools.partial(self._upload_one_part, filename, bucket, key, upload_id, part_size, upload_parts_extra_args, callback)
        for part in executor.map(upload_partial, range(1, num_parts + 1)):
            parts.append(part)
    return parts