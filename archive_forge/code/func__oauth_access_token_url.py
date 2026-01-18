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
def _oauth_access_token_url(self, request_token: Dict[str, Any]) -> str:
    consumer_token = self._oauth_consumer_token()
    url = self._OAUTH_ACCESS_TOKEN_URL
    args = dict(oauth_consumer_key=escape.to_basestring(consumer_token['key']), oauth_token=escape.to_basestring(request_token['key']), oauth_signature_method='HMAC-SHA1', oauth_timestamp=str(int(time.time())), oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)), oauth_version='1.0')
    if 'verifier' in request_token:
        args['oauth_verifier'] = request_token['verifier']
    if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
        signature = _oauth10a_signature(consumer_token, 'GET', url, args, request_token)
    else:
        signature = _oauth_signature(consumer_token, 'GET', url, args, request_token)
    args['oauth_signature'] = signature
    return url + '?' + urllib.parse.urlencode(args)