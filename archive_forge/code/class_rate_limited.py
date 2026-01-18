import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
class rate_limited:
    """
    Decorator for retrying Vultr calls that are rate-limited.

    :param int sleep: Seconds to sleep after being rate-limited.
    :param int retries: Number of retries.
    """

    def __init__(self, sleep=0.5, retries=1):
        self.sleep = sleep
        self.retries = retries

    def __call__(self, call):
        """
        Run ``call`` method until it's not rate-limited.

        The method is invoked while it returns 503 Service Unavailable or the
        allowed number of retries is reached.

        :param callable call: Method to be decorated.
        """

        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(self.retries + 1):
                try:
                    return call(*args, **kwargs)
                except ServiceUnavailableError as e:
                    last_exception = e
                    time.sleep(self.sleep)
            if last_exception:
                raise last_exception
        update_wrapper(wrapper, call)
        return wrapper