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
def _download_file(self, bucket, key, filename, object_size, extra_args, callback):
    if object_size >= self._config.multipart_threshold:
        self._ranged_download(bucket, key, filename, object_size, extra_args, callback)
    else:
        self._get_object(bucket, key, filename, extra_args, callback)