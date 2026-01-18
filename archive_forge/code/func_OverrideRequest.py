from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import copy
import logging
import re
import socket
import types
import six
from six.moves import http_client
from six.moves import urllib
from six.moves import cStringIO
from apitools.base.py import exceptions as apitools_exceptions
from gslib.cloud_api import BadRequestException
from gslib.lazy_wrapper import LazyWrapper
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.utils.constants import DEBUGLEVEL_DUMP_REQUESTS
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
from gslib.utils import text_util
import httplib2
from httplib2 import parse_uri
def OverrideRequest(self, conn, host, absolute_uri, request_uri, method, body, headers, redirections, cachekey):
    """Do the actual request using the connection object.
    Also follow one level of redirects if necessary.
    """
    auths = [(auth.depth(request_uri), auth) for auth in self.authorizations if auth.inscope(host, request_uri)]
    auth = auths and sorted(auths)[0][1] or None
    if auth:
        auth.request(method, request_uri, headers, body)
    response, content = self._conn_request(conn, request_uri, method, body, headers)
    if auth:
        if auth.response(response, body):
            auth.request(method, request_uri, headers, body)
            response, content = self._conn_request(conn, request_uri, method, body, headers)
            response._stale_digest = 1
    if response.status == 401:
        for authorization in self._auth_from_challenge(host, request_uri, headers, response, content):
            authorization.request(method, request_uri, headers, body)
            response, content = self._conn_request(conn, request_uri, method, body, headers)
            if response.status != 401:
                self.authorizations.append(authorization)
                authorization.response(response, body)
                break
    if self.follow_all_redirects or method in ['GET', 'HEAD'] or response.status == 303:
        if self.follow_redirects and response.status in [300, 301, 302, 303, 307]:
            if redirections:
                if 'location' not in response and response.status != 300:
                    raise httplib2.RedirectMissingLocation('Redirected but the response is missing a Location: header.', response, content)
                if 'location' in response:
                    location = response['location']
                    scheme, authority, path, query, fragment = parse_uri(location)
                    if authority is None:
                        response['location'] = urllib.parse.urljoin(absolute_uri, location)
                if response.status == 301 and method in ['GET', 'HEAD']:
                    response['-x-permanent-redirect-url'] = response['location']
                    if 'content-location' not in response:
                        response['content-location'] = absolute_uri
                    httplib2._updateCache(headers, response, content, self.cache, cachekey)
                if 'if-none-match' in headers:
                    del headers['if-none-match']
                if 'if-modified-since' in headers:
                    del headers['if-modified-since']
                if 'authorization' in headers and (not self.forward_authorization_headers):
                    del headers['authorization']
                if 'location' in response:
                    location = response['location']
                    old_response = copy.deepcopy(response)
                    if 'content-location' not in old_response:
                        old_response['content-location'] = absolute_uri
                    redirect_method = method
                    if response.status in [302, 303]:
                        redirect_method = 'GET'
                        body = None
                    response, content = self.request(location, redirect_method, body=body, headers=headers, redirections=redirections - 1, connection_type=conn.__class__)
                    response.previous = old_response
            else:
                raise httplib2.RedirectLimit('Redirected more times than redirection_limit allows.', response, content)
        elif response.status in [200, 203] and method in ['GET', 'HEAD']:
            if 'content-location' in response:
                response['content-location'] = absolute_uri
            httplib2._updateCache(headers, response, content, self.cache, cachekey)
    return (response, content)