import sys
import hmac
import time
import uuid
import base64
import hashlib
from libcloud.utils.py3 import ET, b, u, urlquote
from libcloud.utils.xml import findtext
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import MalformedResponseError

        Sign Aliyun requests parameters and get the signature.

        StringToSign = HTTPMethod + '&' +
                       percentEncode('/') + '&' +
                       percentEncode(CanonicalizedQueryString)
        