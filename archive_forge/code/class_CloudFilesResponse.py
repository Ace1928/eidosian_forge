import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
class CloudFilesResponse(Response):
    valid_response_codes = [httplib.NOT_FOUND, httplib.CONFLICT]

    def success(self):
        i = int(self.status)
        return 200 <= i <= 299 or i in self.valid_response_codes

    def parse_body(self):
        if not self.body:
            return None
        if 'content-type' in self.headers:
            key = 'content-type'
        elif 'Content-Type' in self.headers:
            key = 'Content-Type'
        else:
            raise LibcloudError('Missing content-type header')
        content_type = self.headers[key]
        if content_type.find(';') != -1:
            content_type = content_type.split(';')[0]
        if content_type == 'application/json':
            try:
                data = json.loads(self.body)
            except Exception:
                raise MalformedResponseError('Failed to parse JSON', body=self.body, driver=CloudFilesStorageDriver)
        elif content_type == 'text/plain':
            data = self.body
        else:
            data = self.body
        return data