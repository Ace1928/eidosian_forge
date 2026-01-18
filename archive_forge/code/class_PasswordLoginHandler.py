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
class PasswordLoginHandler(GenericLoginHandler):
    _EXTRA_TOKEN_PARAMS = {'grant_type': 'password'}

    def get(self):
        try:
            errormessage = self.get_argument('error')
        except Exception:
            errormessage = ''
        next_url = self.get_argument('next', None)
        if next_url:
            self.set_cookie('next_url', next_url)
        html = self._login_template.render(errormessage=errormessage, PANEL_CDN=CDN_DIST)
        self.write(html)

    async def post(self):
        username = self.get_argument('username', '')
        password = self.get_argument('password', '')
        if config.oauth_redirect_uri:
            redirect_uri = config.oauth_redirect_uri
        else:
            redirect_uri = f'{self.request.protocol}://{self.request.host}{self._login_endpoint}'
        user, _, _, _ = await self._fetch_access_token(client_id=config.oauth_key, redirect_uri=redirect_uri, username=username, password=password)
        if not user:
            return
        self.redirect('/')