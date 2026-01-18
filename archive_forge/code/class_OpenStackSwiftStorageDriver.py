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
class OpenStackSwiftStorageDriver(CloudFilesStorageDriver):
    """
    Storage driver for the OpenStack Swift.
    """
    type = Provider.CLOUDFILES_SWIFT
    name = 'OpenStack Swift'
    connectionCls = OpenStackSwiftConnection

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region=None, **kwargs):
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, **kwargs)