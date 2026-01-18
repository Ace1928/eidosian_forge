import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
class LibcloudBaseConnection:
    """
    Base connection class to inherit from.

    Note: This class should not be instantiated directly.
    """
    session = None
    proxy_scheme = None
    proxy_host = None
    proxy_port = None
    proxy_username = None
    proxy_password = None
    http_proxy_used = False
    ca_cert = None

    def __init__(self):
        self.session = requests.Session()

    def set_http_proxy(self, proxy_url):
        """
        Set a HTTP proxy which will be used with this connection.

        :param proxy_url: Proxy URL (e.g. http://<hostname>:<port> without
                          authentication and
                          http://<username>:<password>@<hostname>:<port> for
                          basic auth authentication information.
        :type proxy_url: ``str``
        """
        result = self._parse_proxy_url(proxy_url=proxy_url)
        scheme = result[0]
        host = result[1]
        port = result[2]
        username = result[3]
        password = result[4]
        self.proxy_scheme = scheme
        self.proxy_host = host
        self.proxy_port = port
        self.proxy_username = username
        self.proxy_password = password
        self.http_proxy_used = True
        self.session.proxies = {'http': proxy_url, 'https': proxy_url}

    def _parse_proxy_url(self, proxy_url):
        """
        Parse and validate a proxy URL.

        :param proxy_url: Proxy URL (e.g. http://hostname:3128)
        :type proxy_url: ``str``

        :rtype: ``tuple`` (``scheme``, ``hostname``, ``port``)
        """
        parsed = urlparse.urlparse(proxy_url)
        if parsed.scheme not in ('http', 'https'):
            raise ValueError('Only http and https proxies are supported')
        if not parsed.hostname or not parsed.port:
            raise ValueError('proxy_url must be in the following format: <scheme>://<proxy host>:<proxy port>')
        proxy_scheme = parsed.scheme
        proxy_host, proxy_port = (parsed.hostname, parsed.port)
        netloc = parsed.netloc
        if '@' in netloc:
            username_password = netloc.split('@', 1)[0]
            split = username_password.split(':', 1)
            if len(split) < 2:
                raise ValueError('URL is in an invalid format')
            proxy_username, proxy_password = (split[0], split[1])
        else:
            proxy_username = None
            proxy_password = None
        return (proxy_scheme, proxy_host, proxy_port, proxy_username, proxy_password)

    def _setup_verify(self):
        self.verify = libcloud.security.VERIFY_SSL_CERT

    def _setup_ca_cert(self, **kwargs):
        ca_certs_path = kwargs.get('ca_cert', libcloud.security.CA_CERTS_PATH)
        if self.verify is False:
            pass
        elif isinstance(ca_certs_path, list):
            msg = 'Providing a list of CA trusts is no longer supported since libcloud 2.0. Using the first element in the list. See http://libcloud.readthedocs.io/en/latest/other/changes_in_2_0.html#providing-a-list-of-ca-trusts-is-no-longer-supported'
            warnings.warn(msg, DeprecationWarning)
            self.ca_cert = ca_certs_path[0]
        else:
            self.ca_cert = ca_certs_path

    def _setup_signing(self, cert_file=None, key_file=None):
        """
        Setup request signing by mounting a signing
        adapter to the session
        """
        self.session.mount('https://', SignedHTTPSAdapter(cert_file, key_file))