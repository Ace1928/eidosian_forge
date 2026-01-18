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
def _on_request_token(self, authorize_url: str, callback_uri: Optional[str], response: httpclient.HTTPResponse) -> None:
    handler = cast(RequestHandler, self)
    request_token = _oauth_parse_response(response.body)
    data = base64.b64encode(escape.utf8(request_token['key'])) + b'|' + base64.b64encode(escape.utf8(request_token['secret']))
    handler.set_cookie('_oauth_request_token', data)
    args = dict(oauth_token=request_token['key'])
    if callback_uri == 'oob':
        handler.finish(authorize_url + '?' + urllib.parse.urlencode(args))
        return
    elif callback_uri:
        args['oauth_callback'] = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
    handler.redirect(authorize_url + '?' + urllib.parse.urlencode(args))