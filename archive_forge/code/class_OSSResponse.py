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
class OSSResponse(XmlResponse):
    namespace = None
    valid_response_codes = [httplib.NOT_FOUND, httplib.CONFLICT, httplib.BAD_REQUEST]

    def success(self):
        i = int(self.status)
        return 200 <= i <= 299 or i in self.valid_response_codes

    def parse_body(self):
        """
        OSSResponse body is in utf-8 encoding.
        """
        if len(self.body) == 0 and (not self.parse_zero_length_body):
            return self.body
        try:
            parser = ET.XMLParser(encoding='utf-8')
            body = ET.XML(self.body.encode('utf-8'), parser=parser)
        except Exception:
            raise MalformedResponseError('Failed to parse XML', body=self.body, driver=self.connection.driver)
        return body

    def parse_error(self):
        if self.status in [httplib.UNAUTHORIZED, httplib.FORBIDDEN]:
            raise InvalidCredsError(self.body)
        elif self.status == httplib.MOVED_PERMANENTLY:
            raise LibcloudError('This bucket is located in a different ' + 'region. Please use the correct driver.', driver=OSSStorageDriver)
        elif self.status == httplib.METHOD_NOT_ALLOWED:
            raise LibcloudError('The method is not allowed. Status code: %d, headers: %s' % (self.status, self.headers))
        raise LibcloudError('Unknown error. Status code: %d, body: %s' % (self.status, self.body), driver=OSSStorageDriver)