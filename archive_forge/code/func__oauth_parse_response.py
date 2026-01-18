import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def _oauth_parse_response(body: bytes) -> Dict[str, Any]:
    body_str = escape.native_str(body)
    p = urllib.parse.parse_qs(body_str, keep_blank_values=False)
    token = dict(key=p['oauth_token'][0], secret=p['oauth_token_secret'][0])
    special = ('oauth_token', 'oauth_token_secret')
    token.update(((k, p[k][0]) for k in p if k not in special))
    return token