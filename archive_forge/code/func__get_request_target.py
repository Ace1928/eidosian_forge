import logging
import os
import os.path
import socket
import sys
import warnings
from base64 import b64encode
from urllib3 import PoolManager, Timeout, proxy_from_url
from urllib3.exceptions import (
from urllib3.exceptions import (
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import (
from urllib3.util.url import parse_url
import botocore.awsrequest
from botocore.compat import (
from botocore.exceptions import (
def _get_request_target(self, url, proxy_url):
    has_proxy = proxy_url is not None
    if not has_proxy:
        return self._path_url(url)
    proxy_scheme = urlparse(proxy_url).scheme
    using_https_forwarding_proxy = proxy_scheme == 'https' and self._proxies_kwargs().get('use_forwarding_for_https', False)
    if using_https_forwarding_proxy or url.startswith('http:'):
        return url
    else:
        return self._path_url(url)