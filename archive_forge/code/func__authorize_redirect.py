import asyncio
import base64
import codecs
import datetime as dt
import hashlib
import json
import logging
import os
import re
import urllib.parse as urlparse
import uuid
from base64 import urlsafe_b64encode
from functools import partial
import tornado
from bokeh.server.auth_provider import AuthProvider
from tornado.auth import OAuth2Mixin
from tornado.httpclient import HTTPError as HTTPClientError, HTTPRequest
from tornado.web import HTTPError, RequestHandler, decode_signed_value
from tornado.websocket import WebSocketHandler
from .config import config
from .entry_points import entry_points_for
from .io.resources import (
from .io.state import state
from .util import base64url_encode, decode_token
def _authorize_redirect(self, redirect_uri):
    state = self.get_state()
    self.set_state_cookie(state)
    code_verifier, code_challenge = self.get_code()
    self.set_code_cookie(code_verifier)
    params = {'client_id': config.oauth_key, 'response_type': 'code', 'scope': ' '.join(self._SCOPE), 'state': state, 'response_mode': 'query', 'code_challenge': code_challenge, 'code_challenge_method': 'S256', 'redirect_uri': redirect_uri}
    query_params = urlparse.urlencode(params)
    self.redirect(f'{self._OAUTH_AUTHORIZE_URL}?{query_params}')