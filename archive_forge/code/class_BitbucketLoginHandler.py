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
class BitbucketLoginHandler(OAuthLoginHandler):
    _API_BASE_HEADERS = {'Accept': 'application/json'}
    _OAUTH_ACCESS_TOKEN_URL = 'https://bitbucket.org/site/oauth2/access_token'
    _OAUTH_AUTHORIZE_URL = 'https://bitbucket.org/site/oauth2/authorize'
    _OAUTH_USER_URL = 'https://api.bitbucket.org/2.0/user?access_token='
    _USER_KEY = 'username'