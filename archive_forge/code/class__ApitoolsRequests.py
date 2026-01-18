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
class _ApitoolsRequests:
    """A httplib2.Http-like object for use by apitools."""

    def __init__(self, session, response_handler=None, response_encoding=None):
        self.session = session
        self.connections = {}
        if response_handler:
            if not isinstance(response_handler, ResponseHandler):
                raise ValueError('response_handler should be of type ResponseHandler.')
        self._response_handler = response_handler
        self._response_encoding = response_encoding

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

    def request(self, uri, method='GET', body=None, headers=None, redirections=0, connection_type=None):
        """Makes an HTTP request using httplib2 semantics."""
        del connection_type
        if redirections > 0:
            self.session.max_redirects = redirections
        hooks = {}
        if self._response_handler is not None:
            hooks['response'] = self.ResponseHook
            use_stream = self._response_handler.use_stream
        else:
            use_stream = False
        response = self.session.request(method, uri, data=body, headers=headers, stream=use_stream, hooks=hooks)
        headers = dict(response.headers)
        headers['status'] = response.status_code
        if use_stream:
            content = b''
        elif self._response_encoding is not None:
            response.encoding = self._response_encoding
            content = response.text
        else:
            content = response.content
        return (httplib2.Response(headers), content)