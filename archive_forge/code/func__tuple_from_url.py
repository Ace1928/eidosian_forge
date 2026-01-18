import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def _tuple_from_url(self, url):
    secure = 1
    port = None
    scheme, netloc, request_path, param, query, fragment = urlparse.urlparse(url)
    if scheme not in ['http', 'https']:
        raise LibcloudError('Invalid scheme: {} in url {}'.format(scheme, url))
    if scheme == 'http':
        secure = 0
    if ':' in netloc:
        netloc, port = netloc.rsplit(':')
        port = int(port)
    if not port:
        if scheme == 'http':
            port = 80
        else:
            port = 443
    host = netloc
    port = int(port)
    return (host, port, secure, request_path)