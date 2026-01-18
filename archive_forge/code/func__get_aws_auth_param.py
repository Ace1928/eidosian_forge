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
def _get_aws_auth_param(self, params, secret_key, path='/'):
    """
        Creates the signature required for AWS, per
        http://bit.ly/aR7GaQ [docs.amazonwebservices.com]:

        StringToSign = HTTPVerb + "
" +
                       ValueOfHostHeaderInLowercase + "
" +
                       HTTPRequestURI + "
" +
                       CanonicalizedQueryString <from the preceding step>
        """
    connection = self.connection
    keys = list(params.keys())
    keys.sort()
    pairs = []
    for key in keys:
        value = str(params[key])
        pairs.append(urlquote(key, safe='') + '=' + urlquote(value, safe='-_~'))
    qs = '&'.join(pairs)
    hostname = connection.host
    if connection.secure and connection.port != 443 or (not connection.secure and connection.port != 80):
        hostname += ':' + str(connection.port)
    string_to_sign = '\n'.join(('GET', hostname, path, qs))
    b64_hmac = base64.b64encode(hmac.new(b(secret_key), b(string_to_sign), digestmod=sha256).digest())
    return b64_hmac.decode('utf-8')