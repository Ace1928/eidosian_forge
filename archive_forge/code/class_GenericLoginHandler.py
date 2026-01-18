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
class GenericLoginHandler(OAuthLoginHandler):
    _access_token_header = 'Bearer {}'
    _EXTRA_TOKEN_PARAMS = {'grant_type': 'authorization_code'}

    @property
    def _OAUTH_ACCESS_TOKEN_URL(self):
        return config.oauth_extra_params.get('TOKEN_URL', os.environ.get('PANEL_OAUTH_TOKEN_URL'))

    @property
    def _OAUTH_AUTHORIZE_URL(self):
        return config.oauth_extra_params.get('AUTHORIZE_URL', os.environ.get('PANEL_OAUTH_AUTHORIZE_URL'))

    @property
    def _OAUTH_USER_URL(self):
        return config.oauth_extra_params.get('USER_URL', os.environ.get('PANEL_OAUTH_USER_URL'))

    @property
    def _USER_KEY(self):
        return config.oauth_extra_params.get('USER_KEY', os.environ.get('PANEL_USER_KEY', 'email'))