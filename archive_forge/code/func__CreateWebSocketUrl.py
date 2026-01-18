from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import struct
import sys
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
import six
from six.moves.urllib import parse
import socks
def _CreateWebSocketUrl(endpoint, url_query_pieces, url_override):
    """Create URL for WebSocket connection."""
    scheme = URL_SCHEME
    use_mtls = bool(context_aware.Config())
    hostname = MTLS_URL_HOST if use_mtls else URL_HOST
    path_root = URL_PATH_ROOT
    if url_override:
        url_override_parts = parse.urlparse(url_override)
        scheme, hostname, path_override = url_override_parts[:3]
        if path_override and path_override != '/':
            path_root = path_override
    qs = parse.urlencode(url_query_pieces)
    path = '%s%s' % (path_root, endpoint) if path_root.endswith('/') else '%s/%s' % (path_root, endpoint)
    return parse.urlunparse((scheme, hostname, path, '', qs, ''))