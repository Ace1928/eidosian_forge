import collections
import contextlib
import logging
import socket
import time
import httplib2
import six
from six.moves import http_client
from six.moves.urllib import parse
from apitools.base.py import exceptions
from apitools.base.py import util
def _MakeRequestNoRetry(http, http_request, redirections=5, check_response_func=CheckResponse):
    """Send http_request via the given http.

    This wrapper exists to handle translation between the plain httplib2
    request/response types and the Request and Response types above.

    Args:
      http: An httplib2.Http instance, or a http multiplexer that delegates to
          an underlying http, for example, HTTPMultiplexer.
      http_request: A Request to send.
      redirections: (int, default 5) Number of redirects to follow.
      check_response_func: Function to validate the HTTP response.
          Arguments are (Response, response content, url).

    Returns:
      A Response object.

    Raises:
      RequestError if no response could be parsed.

    """
    connection_type = None
    if getattr(http, 'connections', None):
        url_scheme = parse.urlsplit(http_request.url).scheme
        if url_scheme and url_scheme in http.connections:
            connection_type = http.connections[url_scheme]
    new_debuglevel = 4 if httplib2.debuglevel == 4 else 0
    with _Httplib2Debuglevel(http_request, new_debuglevel, http=http):
        info, content = http.request(str(http_request.url), method=str(http_request.http_method), body=http_request.body, headers=http_request.headers, redirections=redirections, connection_type=connection_type)
    if info is None:
        raise exceptions.RequestError()
    response = Response(info, content, http_request.url)
    check_response_func(response)
    return response