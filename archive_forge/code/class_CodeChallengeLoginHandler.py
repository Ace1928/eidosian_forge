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
class CodeChallengeLoginHandler(GenericLoginHandler):

    async def get(self):
        code = self.get_argument('code', '')
        url_state = self.get_argument('state', '')
        if config.oauth_redirect_uri:
            redirect_uri = config.oauth_redirect_uri
        else:
            redirect_uri = f'{self.request.protocol}://{self.request.host}{self._login_endpoint}'
        if not code or not url_state:
            self._authorize_redirect(redirect_uri)
            return
        cookie_state = self.get_state_cookie()
        if cookie_state != url_state:
            log.warning('OAuth state mismatch: %s != %s', cookie_state, url_state)
            raise HTTPError(400, 'OAuth state mismatch')
        state = _deserialize_state(url_state)
        user = await self.get_authenticated_user(redirect_uri, config.oauth_key, url_state, code=code)
        if user is None:
            raise HTTPError(403)
        log.debug('%s authorized user, redirecting to app.', type(self).__name__)
        self.redirect(state.get('next_url', '/'))

    def _authorize_redirect(self, redirect_uri):
        state = self.get_state()
        self.set_state_cookie(state)
        code_verifier, code_challenge = self.get_code()
        self.set_code_cookie(code_verifier)
        params = {'client_id': config.oauth_key, 'response_type': 'code', 'scope': ' '.join(self._SCOPE), 'state': state, 'response_mode': 'query', 'code_challenge': code_challenge, 'code_challenge_method': 'S256', 'redirect_uri': redirect_uri}
        query_params = urlparse.urlencode(params)
        self.redirect(f'{self._OAUTH_AUTHORIZE_URL}?{query_params}')