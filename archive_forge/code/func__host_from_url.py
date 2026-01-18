import base64
import calendar
import datetime
import functools
import hmac
import json
import logging
import time
from collections.abc import Mapping
from email.utils import formatdate
from hashlib import sha1, sha256
from operator import itemgetter
from botocore.compat import (
from botocore.exceptions import NoAuthTokenError, NoCredentialsError
from botocore.utils import (
from botocore.compat import MD5_AVAILABLE  # noqa
def _host_from_url(url):
    url_parts = urlsplit(url)
    host = url_parts.hostname
    if is_valid_ipv6_endpoint_url(url):
        host = f'[{host}]'
    default_ports = {'http': 80, 'https': 443}
    if url_parts.port is not None:
        if url_parts.port != default_ports.get(url_parts.scheme):
            host = '%s:%d' % (host, url_parts.port)
    return host