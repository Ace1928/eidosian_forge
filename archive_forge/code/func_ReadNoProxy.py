from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
from six.moves import urllib
def ReadNoProxy(uri, timeout):
    """Opens a URI with metadata headers, without a proxy, and reads all data.."""
    request = urllib.request.Request(uri, headers=GOOGLE_GCE_METADATA_HEADERS)
    result = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout).read()
    return http_encoding.Decode(result)