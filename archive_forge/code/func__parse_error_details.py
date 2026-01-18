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
def _parse_error_details(self, element):
    """
        Parse code and message from the provided error element.

        :return: ``tuple`` with two elements: (code, message)
        :rtype: ``tuple``
        """
    code = findtext_ignore_namespace(element=element, xpath='Code', namespace=self.namespace)
    message = findtext_ignore_namespace(element=element, xpath='Message', namespace=self.namespace)
    return (code, message)