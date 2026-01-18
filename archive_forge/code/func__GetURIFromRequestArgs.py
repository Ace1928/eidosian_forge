from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import inspect
import io
from google.auth.transport import requests as google_auth_requests
from google.auth.transport.requests import _MutualTlsOffloadAdapter
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import httplib2
import requests
import six
from six.moves import http_client as httplib
from six.moves import urllib
import socks
from urllib3.util.ssl_ import create_urllib3_context
def _GetURIFromRequestArgs(url, params):
    """Gets the complete URI by merging url and params from the request args."""
    url_parts = urllib.parse.urlsplit(url)
    query_params = urllib.parse.parse_qs(url_parts.query, keep_blank_values=True)
    for param, value in six.iteritems(params or {}):
        query_params[param] = value
    url_parts = list(url_parts)
    url_parts[3] = urllib.parse.urlencode(query_params, doseq=True)
    return urllib.parse.urlunsplit(url_parts)