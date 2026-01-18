import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
class LibcloudConnection(LibcloudBaseConnection):
    timeout = None
    host = None
    response = None

    def __init__(self, host, port, secure=None, **kwargs):
        scheme = 'https' if secure is not None and secure else 'http'
        self.host = '{}://{}{}'.format('https' if port == 443 else scheme, host, ':{}'.format(port) if port not in (80, 443) else '')
        https_proxy_url_env = os.environ.get(HTTPS_PROXY_ENV_VARIABLE_NAME, None)
        http_proxy_url_env = os.environ.get(HTTP_PROXY_ENV_VARIABLE_NAME, https_proxy_url_env)
        proxy_url = kwargs.pop('proxy_url', http_proxy_url_env)
        self._setup_verify()
        self._setup_ca_cert()
        LibcloudBaseConnection.__init__(self)
        self.session.timeout = kwargs.pop('timeout', DEFAULT_REQUEST_TIMEOUT)
        if 'cert_file' in kwargs or 'key_file' in kwargs:
            self._setup_signing(**kwargs)
        if proxy_url:
            self.set_http_proxy(proxy_url=proxy_url)

    @property
    def verification(self):
        """
        The option for SSL verification given to underlying requests
        """
        return self.ca_cert if self.ca_cert is not None else self.verify

    def request(self, method, url, body=None, headers=None, raw=False, stream=False, hooks=None):
        url = urlparse.urljoin(self.host, url)
        headers = self._normalize_headers(headers=headers)
        self.response = self.session.request(method=method.lower(), url=url, data=body, headers=headers, allow_redirects=ALLOW_REDIRECTS, stream=stream, verify=self.verification, timeout=self.session.timeout, hooks=hooks)

    def prepared_request(self, method, url, body=None, headers=None, raw=False, stream=False):
        headers = self._normalize_headers(headers=headers)
        req = requests.Request(method, ''.join([self.host, url]), data=body, headers=headers)
        prepped = self.session.prepare_request(req)
        self.response = self.session.send(prepped, stream=stream, verify=self.ca_cert if self.ca_cert is not None else self.verify)

    def getresponse(self):
        return self.response

    def getheaders(self):
        if 'content-encoding' in self.response.headers:
            del self.response.headers['content-encoding']
        return self.response.headers

    @property
    def status(self):
        return self.response.status_code

    @property
    def reason(self):
        return None if self.response.status_code > 400 else self.response.text

    def connect(self):
        pass

    def read(self):
        return self.response.content

    def close(self):
        self.response.close()

    def _normalize_headers(self, headers):
        headers = headers or {}
        for key, value in headers.items():
            if isinstance(value, (int, float)):
                headers[key] = str(value)
        return headers