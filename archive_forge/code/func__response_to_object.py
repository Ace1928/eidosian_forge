import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _response_to_object(self, object_name, container, response):
    """
        Converts a HTTP response to an object (from headers)

        :param object_name: Name of the object
        :type object_name: ``str``

        :param container: Instance of the container holding the blob
        :type: :class:`Container`

        :param response: HTTP Response
        :type node: L{}

        :return: An object instance
        :rtype: :class:`Object`
        """
    headers = response.headers
    size = int(headers['content-length'])
    etag = headers['etag']
    scheme = 'https' if self.secure else 'http'
    extra = {'url': '{}://{}{}'.format(scheme, response.connection.host, response.connection.action), 'etag': etag, 'md5_hash': headers.get('content-md5', None), 'content_type': headers.get('content-type', None), 'content_language': headers.get('content-language', None), 'content_encoding': headers.get('content-encoding', None), 'last_modified': headers['last-modified'], 'lease': {'status': headers.get('x-ms-lease-status', None), 'state': headers.get('x-ms-lease-state', None), 'duration': headers.get('x-ms-lease-duration', None)}, 'blob_type': headers['x-ms-blob-type']}
    if extra['md5_hash']:
        value = binascii.hexlify(base64.b64decode(b(extra['md5_hash'])))
        value = value.decode('ascii')
        extra['md5_hash'] = value
    meta_data = {}
    for key, value in response.headers.items():
        if key.startswith('x-ms-meta-'):
            key = key.split('x-ms-meta-')[1]
            meta_data[key] = value
    return Object(name=object_name, size=size, hash=etag, extra=extra, meta_data=meta_data, container=container, driver=self)