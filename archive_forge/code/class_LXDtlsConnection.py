import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
class LXDtlsConnection(KeyCertificateConnection):
    responseCls = LXDResponse

    def __init__(self, key, secret, secure=True, host='localhost', port=8443, ca_cert='', key_file=None, cert_file=None, certificate_validator=None, **kwargs):
        if certificate_validator is not None:
            certificate_validator(key_file=key_file, cert_file=cert_file)
        else:
            check_certificates(key_file=key_file, cert_file=cert_file, **kwargs)
        super().__init__(key_file=key_file, cert_file=cert_file, secure=secure, host=host, port=port, url=None, proxy_url=None, timeout=None, backoff=None, retry_delay=None)
        self.key_file = key_file
        self.cert_file = cert_file

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'application/json'
        return headers