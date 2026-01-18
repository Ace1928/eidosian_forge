import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class DigestAuthHandler(AbstractAuthHandler):
    """A custom digest authentication handler."""
    scheme = 'digest'
    handler_order = 490

    def auth_params_reusable(self, auth):
        return auth.get('scheme', None) == 'digest'

    def auth_match(self, header, auth):
        scheme, raw_auth = self._parse_auth_header(header)
        if scheme != self.scheme:
            return False
        req_auth = urllib.request.parse_keqv_list(urllib.request.parse_http_list(raw_auth))
        qop = req_auth.get('qop', None)
        if qop != 'auth':
            return False
        H, KD = get_digest_algorithm_impls(req_auth.get('algorithm', 'MD5'))
        if H is None:
            return False
        realm = req_auth.get('realm', None)
        self.update_auth(auth, 'scheme', scheme)
        self.update_auth(auth, 'realm', realm)
        if auth.get('user', None) is None or auth.get('password', None) is None:
            user, password = self.get_user_password(auth)
            self.update_auth(auth, 'user', user)
            self.update_auth(auth, 'password', password)
        try:
            if req_auth.get('algorithm', None) is not None:
                self.update_auth(auth, 'algorithm', req_auth.get('algorithm'))
            nonce = req_auth['nonce']
            if auth.get('nonce', None) != nonce:
                self.update_auth(auth, 'nonce_count', 0)
            self.update_auth(auth, 'nonce', nonce)
            self.update_auth(auth, 'qop', qop)
            auth['opaque'] = req_auth.get('opaque', None)
        except KeyError:
            return False
        return True

    def build_auth_header(self, auth, request):
        uri = urlparse(request.selector).path
        A1 = ('%s:%s:%s' % (auth['user'], auth['realm'], auth['password'])).encode('utf-8')
        A2 = '{}:{}'.format(request.get_method(), uri).encode('utf-8')
        nonce = auth['nonce']
        qop = auth['qop']
        nonce_count = auth['nonce_count'] + 1
        ncvalue = '%08x' % nonce_count
        cnonce = get_new_cnonce(nonce, nonce_count)
        H, KD = get_digest_algorithm_impls(auth.get('algorithm', 'MD5'))
        nonce_data = '{}:{}:{}:{}:{}'.format(nonce, ncvalue, cnonce, qop, H(A2))
        request_digest = KD(H(A1), nonce_data)
        header = 'Digest '
        header += 'username="{}", realm="{}", nonce="{}"'.format(auth['user'], auth['realm'], nonce)
        header += ', uri="%s"' % uri
        header += ', cnonce="{}", nc={}'.format(cnonce, ncvalue)
        header += ', qop="%s"' % qop
        header += ', response="%s"' % request_digest
        opaque = auth.get('opaque', None)
        if opaque:
            header += ', opaque="%s"' % opaque
        if auth.get('algorithm', None):
            header += ', algorithm="%s"' % auth.get('algorithm')
        auth['nonce_count'] = nonce_count
        return header