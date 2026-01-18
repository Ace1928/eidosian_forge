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
class AtmosResponse(XmlResponse):

    def success(self):
        return self.status in (httplib.OK, httplib.CREATED, httplib.NO_CONTENT, httplib.PARTIAL_CONTENT)

    def parse_error(self):
        tree = self.parse_body()
        if tree is None:
            return None
        code = int(tree.find('Code').text)
        message = tree.find('Message').text
        raise AtmosError(code=code, message=message, driver=self.connection.driver)