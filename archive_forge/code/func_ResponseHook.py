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
def ResponseHook(self, response, *args, **kwargs):
    """Response hook to be used if response_handler has been set."""
    del args, kwargs
    if response.status_code not in (httplib.OK, httplib.PARTIAL_CONTENT):
        log.debug('Skipping response_handler as response is invalid.')
        return
    if self._response_handler.use_stream and properties.VALUES.core.log_http.GetBool() and properties.VALUES.core.log_http_streaming_body.GetBool():
        stream = io.BytesIO(response.content)
    else:
        stream = response.raw
    self._response_handler.handle(stream)