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
def _get_proxy_manager(self, proxy_url):
    if proxy_url not in self._proxy_managers:
        proxy_headers = self._proxy_config.proxy_headers_for(proxy_url)
        proxy_ssl_context = self._setup_proxy_ssl_context(proxy_url)
        proxy_manager_kwargs = self._get_pool_manager_kwargs(proxy_headers=proxy_headers)
        proxy_manager_kwargs.update(self._proxies_kwargs(proxy_ssl_context=proxy_ssl_context))
        proxy_manager = proxy_from_url(proxy_url, **proxy_manager_kwargs)
        proxy_manager.pool_classes_by_scheme = self._pool_classes_by_scheme
        self._proxy_managers[proxy_url] = proxy_manager
    return self._proxy_managers[proxy_url]