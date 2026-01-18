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