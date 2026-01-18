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
def _path_url(self, url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if not path:
        path = '/'
    if parsed_url.query:
        path = path + '?' + parsed_url.query
    return path