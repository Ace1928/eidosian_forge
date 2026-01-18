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
class AWSDriver(BaseDriver):

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=None, token=None, **kwargs):
        self.token = token
        super().__init__(key, secret=secret, secure=secure, host=host, port=port, api_version=api_version, region=region, token=token, **kwargs)

    def _ex_connection_class_kwargs(self):
        kwargs = super()._ex_connection_class_kwargs()
        kwargs['token'] = self.token
        return kwargs