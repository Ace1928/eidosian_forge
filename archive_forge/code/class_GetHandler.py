import base64
import datetime
import json
import platform
import threading
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import pymacaroons
import requests
import macaroonbakery._utils as utils
from macaroonbakery.httpbakery._error import DischargeError
from fixtures import (
from httmock import HTTMock, urlmatch
from six.moves.urllib.parse import parse_qs
from six.moves.urllib.request import Request
class GetHandler(BaseHTTPRequestHandler):
    """A mock HTTP server that serves a GET request"""

    def __init__(self, bakery, auth_location, mutate_error, caveats, version, expiry, *args):
        """
        @param bakery used to check incoming requests and macaroons
        for discharge-required errors.
        @param auth_location holds the location of any 3rd party
        authorizer. If this is not None, a 3rd party caveat will be
        added addressed to this location.
        @param mutate_error if non None, will be called with any
        discharge-required error before responding to the client.
        @param caveats called to get caveats to add to the returned
        macaroon.
        @param version holds the version of the bakery that the
        server will purport to serve.
        @param expiry holds the expiry for the macaroon that will be created
        in _write_discharge_error
        """
        self._bakery = bakery
        self._auth_location = auth_location
        self._mutate_error = mutate_error
        self._caveats = caveats
        self._server_version = version
        self._expiry = expiry
        BaseHTTPRequestHandler.__init__(self, *args)

    def do_GET(self):
        """do_GET implements a handler for the HTTP GET method"""
        ctx = checkers.AuthContext()
        auth_checker = self._bakery.checker.auth(httpbakery.extract_macaroons(self.headers))
        try:
            auth_checker.allow(ctx, [TEST_OP])
        except (bakery.PermissionDenied, bakery.VerificationError) as exc:
            return self._write_discharge_error(exc)
        self.send_response(200)
        self.end_headers()
        content_len = int(self.headers.get('content-length', 0))
        content = 'done'
        if self.path != '/no-body' and content_len > 0:
            body = self.rfile.read(content_len)
            content = content + ' ' + body
        self.wfile.write(content.encode('utf-8'))
        return

    def _write_discharge_error(self, exc):
        version = httpbakery.request_version(self.headers)
        if version < bakery.LATEST_VERSION:
            self._server_version = version
        caveats = []
        if self._auth_location != '':
            caveats = [checkers.Caveat(location=self._auth_location, condition='is-ok')]
        if self._caveats is not None:
            caveats.extend(self._caveats)
        m = self._bakery.oven.macaroon(version=bakery.LATEST_VERSION, expiry=self._expiry, caveats=caveats, ops=[TEST_OP])
        content, headers = httpbakery.discharge_required_response(m, '/', 'test', exc.args[0])
        self.send_response(401)
        for h in headers:
            self.send_header(h, headers[h])
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(content)