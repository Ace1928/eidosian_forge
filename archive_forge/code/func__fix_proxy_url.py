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
def _fix_proxy_url(self, proxy_url):
    if proxy_url.startswith('http:') or proxy_url.startswith('https:'):
        return proxy_url
    elif proxy_url.startswith('//'):
        return 'http:' + proxy_url
    else:
        return 'http://' + proxy_url