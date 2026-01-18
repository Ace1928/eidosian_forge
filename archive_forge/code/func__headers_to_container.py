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
def _headers_to_container(self, name, headers):
    size = int(headers.get('x-container-bytes-used', 0))
    object_count = int(headers.get('x-container-object-count', 0))
    extra = {'object_count': object_count, 'size': size}
    container = Container(name=name, extra=extra, driver=self)
    return container