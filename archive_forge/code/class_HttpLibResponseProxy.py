import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
class HttpLibResponseProxy:
    """
    Provides a proxy pattern around the :class:`requests.Reponse`
    object to a :class:`httplib.HTTPResponse` object
    """

    def __init__(self, response):
        self._response = response

    def read(self, amt=None):
        return self._response.text

    def getheader(self, name, default=None):
        """
        Get the contents of the header name, or default
        if there is no matching header.
        """
        if name in self._response.headers.keys():
            return self._response.headers[name]
        else:
            return default

    def getheaders(self):
        """
        Return a list of (header, value) tuples.
        """
        return list(self._response.headers.items())

    @property
    def status(self):
        return self._response.status_code

    @property
    def reason(self):
        return self._response.reason

    @property
    def version(self):
        return '11'

    @property
    def body(self):
        return self._response.content