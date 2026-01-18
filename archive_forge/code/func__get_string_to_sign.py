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
def _get_string_to_sign(self, params, headers, dt, method, path, data):
    canonical_request = self._get_canonical_request(params=params, headers=headers, method=method, path=path, data=data)
    return '\n'.join(['AWS4-HMAC-SHA256', dt.strftime('%Y%m%dT%H%M%SZ'), self._get_credential_scope(dt), _hash(canonical_request)])