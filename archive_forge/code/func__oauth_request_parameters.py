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
def _oauth_request_parameters(self, url: str, access_token: Dict[str, Any], parameters: Dict[str, Any]={}, method: str='GET') -> Dict[str, Any]:
    """Returns the OAuth parameters as a dict for the given request.

        parameters should include all POST arguments and query string arguments
        that will be sent with the request.
        """
    consumer_token = self._oauth_consumer_token()
    base_args = dict(oauth_consumer_key=escape.to_basestring(consumer_token['key']), oauth_token=escape.to_basestring(access_token['key']), oauth_signature_method='HMAC-SHA1', oauth_timestamp=str(int(time.time())), oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)), oauth_version='1.0')
    args = {}
    args.update(base_args)
    args.update(parameters)
    if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
        signature = _oauth10a_signature(consumer_token, method, url, args, access_token)
    else:
        signature = _oauth_signature(consumer_token, method, url, args, access_token)
    base_args['oauth_signature'] = escape.to_basestring(signature)
    return base_args