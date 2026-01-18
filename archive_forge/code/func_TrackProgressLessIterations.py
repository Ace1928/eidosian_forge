from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import random
import re
import socket
import time
import six
from six.moves import urllib
from six.moves import http_client
from boto import config
from boto import UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from gslib.exception import InvalidUrlError
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.constants import UTF8
def TrackProgressLessIterations(self, service_had_bytes_before_attempt, debug=0):
    """Tracks the number of iterations without progress.

    Performs randomized exponential backoff.

    Args:
      service_had_bytes_before_attempt: Number of bytes the service had prior
                                       to this upload attempt.
      debug: debug level 0..3
    """
    if self.service_has_bytes > service_had_bytes_before_attempt:
        self.progress_less_iterations = 0
    else:
        self.progress_less_iterations += 1
    if self.progress_less_iterations > self.num_retries:
        raise ResumableUploadException('Too many resumable upload attempts failed without progress. You might try this upload again later', ResumableTransferDisposition.ABORT_CUR_PROCESS)
    sleep_time_secs = min(random.random() * 2 ** self.progress_less_iterations, GetMaxRetryDelay())
    if debug >= 1:
        self.logger.debug('Got retryable failure (%d progress-less in a row).\nSleeping %3.1f seconds before re-trying', self.progress_less_iterations, sleep_time_secs)
    time.sleep(sleep_time_secs)