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
def _perform_io_writes(self, filename):
    with self._os.open(filename, 'wb') as f:
        while True:
            task = self._ioqueue.get()
            if task is SHUTDOWN_SENTINEL:
                logger.debug('Shutdown sentinel received in IO handler, shutting down IO handler.')
                return
            else:
                try:
                    offset, data = task
                    f.seek(offset)
                    f.write(data)
                except Exception as e:
                    logger.debug('Caught exception in IO thread: %s', e, exc_info=True)
                    self._ioqueue.trigger_shutdown()
                    raise