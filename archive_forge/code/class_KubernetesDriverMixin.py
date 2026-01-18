import os
import base64
import warnings
from typing import Optional
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import (
from libcloud.common.types import InvalidCredsError
class KubernetesDriverMixin:
    """
    Base driver class to be used with various Kubernetes drivers.

    NOTE: This base class can be used in different APIs such as container and
    compute one.
    """

    def __init__(self, key=None, secret=None, secure=False, host='localhost', port=4243, key_file=None, cert_file=None, ca_cert=None, ex_token_bearer_auth=False):
        """
        :param    key: API key or username to be used (required)
        :type     key: ``str``

        :param    secret: Secret password to be used (required)
        :type     secret: ``str``

        :param    secure: Whether to use HTTPS or HTTP. Note: Some providers
                          only support HTTPS, and it is on by default.
        :type     secure: ``bool``

        :param    host: Override hostname used for connections.
        :type     host: ``str``

        :param    port: Override port used for connections.
        :type     port: ``int``

        :param    key_file: Path to the key file used to authenticate (when
                            using key file auth).
        :type     key_file: ``str``

        :param    cert_file: Path to the cert file used to authenticate (when
                             using key file auth).
        :type     cert_file: ``str``

        :param    ex_token_bearer_auth: True to use token bearer auth.
        :type     ex_token_bearer_auth: ``bool``

        :return: ``None``
        """
        if ex_token_bearer_auth:
            self.connectionCls = KubernetesTokenAuthConnection
            if not key:
                msg = 'The token must be a string provided via "key" argument'
                raise ValueError(msg)
            secure = True
        if key_file or cert_file:
            if not (key_file and cert_file):
                raise ValueError('Both key and certificate files are needed')
        if key_file:
            self.connectionCls = KubernetesTLSAuthConnection
            self.key_file = key_file
            self.cert_file = cert_file
            secure = True
        if host and host.startswith('https://'):
            secure = True
        host = self._santize_host(host=host)
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, key_file=key_file, cert_file=cert_file)
        if ca_cert:
            self.connection.connection.ca_cert = ca_cert
        else:
            warnings.warn("Kubernetes has its own CA, since you didn't supply a CA certificate be aware that SSL verification will be disabled for this session.")
            self.connection.connection.ca_cert = False
        self.connection.secure = secure
        self.connection.host = host
        if port is not None:
            self.connection.port = port

    def _ex_connection_class_kwargs(self):
        kwargs = {}
        if hasattr(self, 'key_file'):
            kwargs['key_file'] = self.key_file
        if hasattr(self, 'cert_file'):
            kwargs['cert_file'] = self.cert_file
        return kwargs

    def _santize_host(self, host=None):
        """
        Sanitize "host" argument any remove any protocol prefix (if specified).
        """
        if not host:
            return None
        prefixes = ['http://', 'https://']
        for prefix in prefixes:
            if host.startswith(prefix):
                host = host.lstrip(prefix)
        return host