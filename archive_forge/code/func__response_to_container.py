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
def _response_to_container(self, container_name, response):
    """
        Converts a HTTP response to a container instance

        :param container_name: Name of the container
        :type container_name: ``str``

        :param response: HTTP Response
        :type node: L{}

        :return: A container instance
        :rtype: :class:`Container`
        """
    headers = response.headers
    scheme = 'https' if self.secure else 'http'
    extra = {'url': '{}://{}{}'.format(scheme, response.connection.host, response.connection.action), 'etag': headers['etag'], 'last_modified': headers['last-modified'], 'lease': {'status': headers.get('x-ms-lease-status', None), 'state': headers.get('x-ms-lease-state', None), 'duration': headers.get('x-ms-lease-duration', None)}, 'meta_data': {}}
    for key, value in response.headers.items():
        if key.startswith('x-ms-meta-'):
            key = key.split('x-ms-meta-')[1]
            extra['meta_data'][key] = value
    return Container(name=container_name, extra=extra, driver=self)