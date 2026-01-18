from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import os
import threading
import time
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import CloudApi
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
@contextlib.contextmanager
def AcquireLockWithTimeout(lock, timeout):
    result = lock.acquire(timeout=timeout)
    yield result
    if result:
        lock.release()