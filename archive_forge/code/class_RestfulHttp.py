import atexit
import errno
import os
import re
import shutil
import sys
import tempfile
from hashlib import md5
from io import BytesIO
from json import dumps
from time import sleep
from httplib2 import Http, urlnorm
from wadllib.application import Application
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError, error_for
from lazr.uri import URI
class RestfulHttp(Http):
    """An Http subclass with some custom behavior.

    This Http client uses the TE header instead of the Accept-Encoding
    header to ask for compressed representations. It also knows how to
    react when its cache is a MultipleRepresentationCache.
    """
    maximum_cache_filename_length = 143

    def __init__(self, authorizer=None, cache=None, timeout=None, proxy_info=proxy_info_from_environment):
        cert_disabled = ssl_certificate_validation_disabled()
        super(RestfulHttp, self).__init__(cache, timeout, proxy_info, disable_ssl_certificate_validation=cert_disabled, ca_certs=SYSTEM_CA_CERTS)
        self.authorizer = authorizer
        if self.authorizer is not None:
            self.authorizer.authorizeSession(self)

    def _request(self, conn, host, absolute_uri, request_uri, method, body, headers, redirections, cachekey):
        """Use the authorizer to authorize an outgoing request."""
        if 'authorization' in headers:
            del headers['authorization']
        if self.authorizer is not None:
            self.authorizer.authorizeRequest(absolute_uri, method, body, headers)
        return super(RestfulHttp, self)._request(conn, host, absolute_uri, request_uri, method, body, headers, redirections, cachekey)

    def _getCachedHeader(self, uri, header):
        """Retrieve a cached value for an HTTP header."""
        if isinstance(self.cache, MultipleRepresentationCache):
            return self.cache._getCachedHeader(uri, header)
        return None