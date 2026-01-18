import hmac
import time
import base64
import hashlib
from io import FileIO as file
from libcloud.utils.py3 import b, next, httplib, urlparse, urlquote, urlencode, urlunquote
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError
from libcloud.storage.base import CHUNK_SIZE, Object, Container, StorageDriver
from libcloud.storage.types import (
def _cdn_signature(self, path, params, expiry):
    key = base64.b64decode(self.secret)
    signature = '\n'.join(['GET', path.lower(), self.key, expiry])
    signature = hmac.new(key, signature, hashlib.sha1).digest()
    return base64.b64encode(signature)