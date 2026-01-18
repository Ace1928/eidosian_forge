import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
def _get_http_response_with_retries(method, url, max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status=True, allow_redirects=None, respect_retry_after_header=True, **kwargs):
    """Performs an HTTP request using Python's `requests` module with an automatic retry policy.

    Args:
        method: A string indicating the method to use, e.g. "GET", "POST", "PUT".
        url: The target URL address for the HTTP request.
        max_retries: Maximum total number of retries.
        backoff_factor: A time factor for exponential backoff. e.g. value 5 means the HTTP
            request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
            exponential backoff.
        backoff_jitter: A random jitter to add to the backoff interval.
        retry_codes: A list of HTTP response error codes that qualifies for retry.
        raise_on_status: Whether to raise an exception, or return a response, if status falls
            in retry_codes range and retries have been exhausted.
        kwargs: Additional keyword arguments to pass to `requests.Session.request()`

    Returns:
        requests.Response object.
    """
    session = _get_request_session(max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status, respect_retry_after_header)
    env_value = os.getenv('MLFLOW_ALLOW_HTTP_REDIRECTS', 'true').lower() in ['true', '1']
    allow_redirects = env_value if allow_redirects is None else allow_redirects
    return session.request(method, url, allow_redirects=allow_redirects, **kwargs)