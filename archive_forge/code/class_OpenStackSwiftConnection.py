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
class OpenStackSwiftConnection(OpenStackBaseConnection):
    """
    Connection class for the OpenStack Swift endpoint.
    """
    responseCls = CloudFilesResponse
    rawResponseCls = CloudFilesRawResponse
    auth_url = AUTH_URL
    _auth_version = '1.0'

    def __init__(self, user_id, key, secure=True, **kwargs):
        kwargs.pop('use_internal_url', None)
        super().__init__(user_id, key, secure=secure, **kwargs)
        self.api_version = API_VERSION
        self.accept_format = 'application/json'
        self._service_type = self._ex_force_service_type or 'object-store'
        self._service_name = self._ex_force_service_name or 'swift'
        if self._ex_force_service_region:
            self._service_region = self._ex_force_service_region
        else:
            self._service_region = None

    def get_endpoint(self, *args, **kwargs):
        if '2.0' in self._auth_version or '3.x' in self._auth_version:
            endpoint = self.service_catalog.get_endpoint(service_type=self._service_type, name=self._service_name, region=self._service_region)
        elif '1.1' in self._auth_version or '1.0' in self._auth_version:
            endpoint = self.service_catalog.get_endpoint(name=self._service_name, region=self._service_region)
        else:
            endpoint = None
        if endpoint:
            return endpoint.url
        else:
            raise LibcloudError('Could not find specified endpoint')

    def request(self, action, params=None, data='', headers=None, method='GET', raw=False, cdn_request=False):
        if not headers:
            headers = {}
        if not params:
            params = {}
        self.cdn_request = cdn_request
        params['format'] = 'json'
        if method in ['POST', 'PUT'] and 'Content-Type' not in headers:
            headers.update({'Content-Type': 'application/json; charset=UTF-8'})
        return super().request(action=action, params=params, data=data, method=method, headers=headers, raw=raw)