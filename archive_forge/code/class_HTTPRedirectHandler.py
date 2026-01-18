import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class HTTPRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handles redirect requests.

    We have to implement our own scheme because we use a specific
    Request object and because we want to implement a specific
    policy.
    """
    _debuglevel = DEBUG

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """See urllib.request.HTTPRedirectHandler.redirect_request"""
        origin_req_host = req.origin_req_host
        if code in (301, 302, 303, 307, 308):
            return Request(req.get_method(), newurl, headers=req.headers, origin_req_host=origin_req_host, unverifiable=True, connection=None, parent=req)
        else:
            raise urllib.request.HTTPError(req.get_full_url(), code, msg, headers, fp)

    def http_error_302(self, req, fp, code, msg, headers):
        """Requests the redirected to URI.

        Copied from urllib.request to be able to clean the pipe of the associated
        connection, *before* issuing the redirected request but *after* having
        eventually raised an error.
        """
        if 'location' in headers:
            newurl = headers.get('location')
        elif 'uri' in headers:
            newurl = headers.get('uri')
        else:
            return
        newurl = urljoin(req.get_full_url(), newurl)
        if self._debuglevel >= 1:
            print('Redirected to: {} (followed: {!r})'.format(newurl, req.follow_redirections))
        if req.follow_redirections is False:
            req.redirected_to = newurl
            return fp
        redirected_req = self.redirect_request(req, fp, code, msg, headers, newurl)
        if hasattr(req, 'redirect_dict'):
            visited = redirected_req.redirect_dict = req.redirect_dict
            if visited.get(newurl, 0) >= self.max_repeats or len(visited) >= self.max_redirections:
                raise urllib.request.HTTPError(req.get_full_url(), code, self.inf_msg + msg, headers, fp)
        else:
            visited = redirected_req.redirect_dict = req.redirect_dict = {}
        visited[newurl] = visited.get(newurl, 0) + 1
        fp.close()
        req.connection.cleanup_pipe()
        return self.parent.open(redirected_req)
    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302