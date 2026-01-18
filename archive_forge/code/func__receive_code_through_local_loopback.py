import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
def _receive_code_through_local_loopback(self):
    """
        Start a local HTTP server that listens to a single GET request that is expected to be made
        by the loopback in the sign-in process and stops again afterwards.
        See https://developers.google.com/identity/protocols/oauth2/native-app#redirect-uri_loopback

        :return: The access code that was extracted from the local loopback GET request
        :rtype: ``str``
        """
    access_code = None

    class AccessCodeReceiver(BaseHTTPRequestHandler):

        def do_GET(self_):
            query = urlparse.urlparse(self_.path).query
            query_components = dict((qc.split('=') for qc in query.split('&')))
            if 'state' in query_components and query_components['state'] != urllib.parse.quote(self._state):
                raise ValueError("States do not match: {} != {}, can't trust authentication".format(self._state, query_components['state']))
            nonlocal access_code
            access_code = query_components['code']
            self_.send_response(200)
            self_.send_header('Content-type', 'text/html')
            self_.end_headers()
            self_.wfile.write(b'<html><head><title>Libcloud Sign-In</title></head>')
            self_.wfile.write(b'<body><p>You can now close this tab</p>')
    if '127.0.0.1' in self.redirect_uri or '[::1]' in self.redirect_uri or 'localhost' in self.redirect_uri:
        server_address = ('localhost', self.redirect_uri_port)
    else:
        server_address = (self.redirect_uri, self.redirect_uri_port)
    server = HTTPServer(server_address=server_address, RequestHandlerClass=AccessCodeReceiver)
    server.handle_request()
    if access_code is None:
        raise RuntimeError('Could not receive OAuth2 code: could not extract code though loopback')
    return access_code