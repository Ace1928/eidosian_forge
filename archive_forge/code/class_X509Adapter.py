from OpenSSL.crypto import PKey, X509
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (load_pem_private_key,
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.backends import default_backend
from datetime import datetime
from requests.adapters import HTTPAdapter
import requests
from .. import exceptions as exc
importing the protocol constants from _ssl instead of ssl because only the
class X509Adapter(HTTPAdapter):
    """Adapter for use with X.509 certificates.

    Provides an interface for Requests sessions to contact HTTPS urls and
    authenticate  with an X.509 cert by implementing the Transport Adapter
    interface. This class will need to be manually instantiated and mounted
    to the session

    :param pool_connections: The number of urllib3 connection pools to
           cache.
    :param pool_maxsize: The maximum number of connections to save in the
            pool.
    :param max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups,
        socket connections and connection timeouts, never to requests where
        data has made it to the server. By default, Requests does not retry
        failed connections. If you need granular control over the
        conditions under which we retry a request, import urllib3's
        ``Retry`` class and pass that instead.
    :param pool_block: Whether the connection pool should block for
            connections.

    :param bytes cert_bytes:
        bytes object containing contents of a cryptography.x509Certificate
        object using the encoding specified by the ``encoding`` parameter.
    :param bytes pk_bytes:
        bytes object containing contents of a object that implements
        ``cryptography.hazmat.primitives.serialization.PrivateFormat``
        using the encoding specified by the ``encoding`` parameter.
    :param password:
        string or utf8 encoded bytes containing the passphrase used for the
        private key. None if unencrypted. Defaults to None.
    :param encoding:
        Enumeration detailing the encoding method used on the ``cert_bytes``
        parameter. Can be either PEM or DER. Defaults to PEM.
    :type encoding:
        :class: `cryptography.hazmat.primitives.serialization.Encoding`

    Usage::

      >>> import requests
      >>> from requests_toolbelt.adapters.x509 import X509Adapter
      >>> s = requests.Session()
      >>> a = X509Adapter(max_retries=3,
                cert_bytes=b'...', pk_bytes=b'...', encoding='...'
      >>> s.mount('https://', a)
    """

    def __init__(self, *args, **kwargs):
        self._import_pyopensslcontext()
        self._check_version()
        cert_bytes = kwargs.pop('cert_bytes', None)
        pk_bytes = kwargs.pop('pk_bytes', None)
        password = kwargs.pop('password', None)
        encoding = kwargs.pop('encoding', Encoding.PEM)
        password_bytes = None
        if cert_bytes is None or not isinstance(cert_bytes, bytes):
            raise ValueError('Invalid cert content provided. You must provide an X.509 cert formatted as a byte array.')
        if pk_bytes is None or not isinstance(pk_bytes, bytes):
            raise ValueError('Invalid private key content provided. You must provide a private key formatted as a byte array.')
        if isinstance(password, bytes):
            password_bytes = password
        elif password:
            password_bytes = password.encode('utf8')
        self.ssl_context = create_ssl_context(cert_bytes, pk_bytes, password_bytes, encoding)
        super(X509Adapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self.ssl_context:
            kwargs['ssl_context'] = self.ssl_context
        return super(X509Adapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        if self.ssl_context:
            kwargs['ssl_context'] = self.ssl_context
        return super(X509Adapter, self).proxy_manager_for(*args, **kwargs)

    def _import_pyopensslcontext(self):
        global PyOpenSSLContext
        if requests.__build__ < 135680:
            PyOpenSSLContext = None
        else:
            try:
                from requests.packages.urllib3.contrib.pyopenssl import PyOpenSSLContext
            except ImportError:
                try:
                    from urllib3.contrib.pyopenssl import PyOpenSSLContext
                except ImportError:
                    PyOpenSSLContext = None

    def _check_version(self):
        if PyOpenSSLContext is None:
            raise exc.VersionMismatchError('The X509Adapter requires at least Requests 2.12.0 to be installed. Version {} was found instead.'.format(requests.__version__))