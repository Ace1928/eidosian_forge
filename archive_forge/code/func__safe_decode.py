import os
import hmac
import time
import base64
import codecs
from hashlib import sha1
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _safe_decode(self, encoded):
    """
        Decode it as an escaped string and then treat the content as
        UTF-8 encoded.
        """
    try:
        if encoded:
            unescaped, _ign = codecs.escape_decode(encoded)
            return unescaped.decode('utf-8')
        return encoded
    except Exception:
        return encoded