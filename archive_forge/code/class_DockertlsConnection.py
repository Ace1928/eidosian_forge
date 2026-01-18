import os
import re
import shlex
import base64
import datetime
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
class DockertlsConnection(KeyCertificateConnection):
    responseCls = DockerResponse

    def __init__(self, key, secret, secure=True, host='localhost', port=4243, key_file='', cert_file='', **kwargs):
        super().__init__(key_file=key_file, cert_file=cert_file, secure=secure, host=host, port=port, url=None, proxy_url=None, timeout=None, backoff=None, retry_delay=None)
        if key_file:
            keypath = os.path.expanduser(key_file)
            is_file_path = os.path.exists(keypath) and os.path.isfile(keypath)
            if not is_file_path:
                raise InvalidCredsError('You need an key PEM file to authenticate with Docker tls. This can be found in the server.')
            self.key_file = key_file
            certpath = os.path.expanduser(cert_file)
            is_file_path = os.path.exists(certpath) and os.path.isfile(certpath)
            if not is_file_path:
                raise InvalidCredsError('You need an certificate PEM file to authenticate with Docker tls. This can be found in the server.')
            self.cert_file = cert_file

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'application/json'
        return headers