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
def _validate_all_known_args(self, actual, allowed):
    for kwarg in actual:
        if kwarg not in allowed:
            raise ValueError("Invalid extra_args key '%s', must be one of: %s" % (kwarg, ', '.join(allowed)))