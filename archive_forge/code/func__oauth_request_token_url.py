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
def _oauth_request_token_url(self, redirect_uri: Optional[str]=None, client_id: Optional[str]=None, client_secret: Optional[str]=None, code: Optional[str]=None, extra_params: Optional[Dict[str, Any]]=None) -> str:
    url = self._OAUTH_ACCESS_TOKEN_URL
    args = {}
    if redirect_uri is not None:
        args['redirect_uri'] = redirect_uri
    if code is not None:
        args['code'] = code
    if client_id is not None:
        args['client_id'] = client_id
    if client_secret is not None:
        args['client_secret'] = client_secret
    if extra_params:
        args.update(extra_params)
    return url_concat(url, args)