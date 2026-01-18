import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
@lru_cache(maxsize=64)
def _cached_get_request_session(max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status, _pid, respect_retry_after_header=True):
    """
    This function should not be called directly. Instead, use `_get_request_session` below.
    """
    retry_kwargs = {'total': max_retries, 'connect': max_retries, 'read': max_retries, 'redirect': max_retries, 'status': max_retries, 'status_forcelist': retry_codes, 'backoff_factor': backoff_factor, 'backoff_jitter': backoff_jitter, 'raise_on_status': raise_on_status, 'respect_retry_after_header': respect_retry_after_header}
    urllib3_version = Version(urllib3.__version__)
    if urllib3_version >= Version('1.26.0'):
        retry_kwargs['allowed_methods'] = None
    else:
        retry_kwargs['method_whitelist'] = None
    if urllib3_version < Version('2.0'):
        retry = JitteredRetry(**retry_kwargs)
    else:
        retry = Retry(**retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session