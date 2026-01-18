import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def CalculateWaitForRetry(retry_attempt, max_wait=60):
    """Calculates amount of time to wait before a retry attempt.

    Wait time grows exponentially with the number of attempts. A
    random amount of jitter is added to spread out retry attempts from
    different clients.

    Args:
      retry_attempt: Retry attempt counter.
      max_wait: Upper bound for wait time [seconds].

    Returns:
      Number of seconds to wait before retrying request.

    """
    wait_time = 2 ** retry_attempt
    max_jitter = wait_time / 4.0
    wait_time += random.uniform(-max_jitter, max_jitter)
    return max(1, min(wait_time, max_wait))