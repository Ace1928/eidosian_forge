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
def _upload_object_manifest(self, container, object_name, extra=None, verify_hash=True):
    extra = extra or {}
    meta_data = extra.get('meta_data')
    container_name_encoded = self._encode_container_name(container.name)
    object_name_encoded = self._encode_object_name(object_name)
    request_path = '/{}/{}'.format(container_name_encoded, object_name_encoded)
    headers = {'X-Auth-Token': self.connection.auth_token, 'X-Object-Manifest': '{}/{}/'.format(container_name_encoded, object_name_encoded)}
    data = ''
    response = self.connection.request(request_path, method='PUT', data=data, headers=headers, raw=True)
    object_hash = None
    if verify_hash:
        hash_function = self._get_hash_function()
        hash_function.update(b(data))
        data_hash = hash_function.hexdigest()
        object_hash = response.headers.get('etag')
        if object_hash != data_hash:
            raise ObjectHashMismatchError(value=('MD5 hash checksum does not match (expected=%s, ' + 'actual=%s)') % (data_hash, object_hash), object_name=object_name, driver=self)
    obj = Object(name=object_name, size=0, hash=object_hash, extra=None, meta_data=meta_data, container=container, driver=self)
    return obj