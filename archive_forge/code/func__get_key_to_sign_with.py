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
def _get_key_to_sign_with(self, dt):
    return _sign(_sign(_sign(_sign('AWS4' + self.access_secret, dt.strftime('%Y%m%d')), self.connection.driver.region_name), self.connection.service_name), 'aws4_request')