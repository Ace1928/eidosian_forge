import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
def _map_url(self, url):
    url = url or self.url
    if not url:
        url, _ = self.get_auth()
    scheme, netloc, path, params, query, fragment = urlparse(url)
    if URI_PATTERN_VERSION.search(path):
        path = URI_PATTERN_VERSION.sub('/info', path)
    elif not URI_PATTERN_INFO.search(path):
        if path.endswith('/'):
            path += 'info'
        else:
            path += '/info'
    return urlunparse((scheme, netloc, path, params, query, fragment))