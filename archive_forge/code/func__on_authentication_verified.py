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
def _on_authentication_verified(self, response: httpclient.HTTPResponse) -> Dict[str, Any]:
    handler = cast(RequestHandler, self)
    if b'is_valid:true' not in response.body:
        raise AuthError('Invalid OpenID response: %r' % response.body)
    ax_ns = None
    for key in handler.request.arguments:
        if key.startswith('openid.ns.') and handler.get_argument(key) == 'http://openid.net/srv/ax/1.0':
            ax_ns = key[10:]
            break

    def get_ax_arg(uri: str) -> str:
        if not ax_ns:
            return ''
        prefix = 'openid.' + ax_ns + '.type.'
        ax_name = None
        for name in handler.request.arguments.keys():
            if handler.get_argument(name) == uri and name.startswith(prefix):
                part = name[len(prefix):]
                ax_name = 'openid.' + ax_ns + '.value.' + part
                break
        if not ax_name:
            return ''
        return handler.get_argument(ax_name, '')
    email = get_ax_arg('http://axschema.org/contact/email')
    name = get_ax_arg('http://axschema.org/namePerson')
    first_name = get_ax_arg('http://axschema.org/namePerson/first')
    last_name = get_ax_arg('http://axschema.org/namePerson/last')
    username = get_ax_arg('http://axschema.org/namePerson/friendly')
    locale = get_ax_arg('http://axschema.org/pref/language').lower()
    user = dict()
    name_parts = []
    if first_name:
        user['first_name'] = first_name
        name_parts.append(first_name)
    if last_name:
        user['last_name'] = last_name
        name_parts.append(last_name)
    if name:
        user['name'] = name
    elif name_parts:
        user['name'] = ' '.join(name_parts)
    elif email:
        user['name'] = email.split('@')[0]
    if email:
        user['email'] = email
    if locale:
        user['locale'] = locale
    if username:
        user['username'] = username
    claimed_id = handler.get_argument('openid.claimed_id', None)
    if claimed_id:
        user['claimed_id'] = claimed_id
    return user