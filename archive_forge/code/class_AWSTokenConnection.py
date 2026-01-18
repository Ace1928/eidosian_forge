import hmac
import time
import base64
import hashlib
from typing import Dict, Type, Optional
from hashlib import sha256
from datetime import datetime
from libcloud.utils.py3 import ET, b, httplib, urlquote, basestring, _real_unicode
from libcloud.utils.xml import findall_ignore_namespace, findtext_ignore_namespace
from libcloud.common.base import BaseDriver, XmlResponse, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
class AWSTokenConnection(ConnectionUserAndKey):

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, proxy_url=None, token=None, retry_delay=None, backoff=None):
        self.token = token
        super().__init__(user_id, key, secure=secure, host=host, port=port, url=url, timeout=timeout, retry_delay=retry_delay, backoff=backoff, proxy_url=proxy_url)

    def add_default_params(self, params):
        if self.token:
            params['x-amz-security-token'] = self.token
        return super().add_default_params(params)

    def add_default_headers(self, headers):
        if self.token:
            headers['x-amz-security-token'] = self.token
        return super().add_default_headers(headers)