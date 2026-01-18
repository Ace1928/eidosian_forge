import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class DigestAuthServer(AuthServer):
    """A digest authentication server"""
    auth_nonce = 'now!'

    def __init__(self, request_handler, auth_scheme, protocol_version=None):
        AuthServer.__init__(self, request_handler, auth_scheme, protocol_version=protocol_version)

    def digest_authorized(self, auth, command):
        nonce = auth['nonce']
        if nonce != self.auth_nonce:
            return False
        realm = auth['realm']
        if realm != self.auth_realm:
            return False
        user = auth['username']
        if user not in self.password_of:
            return False
        algorithm = auth['algorithm']
        if algorithm != 'MD5':
            return False
        qop = auth['qop']
        if qop != 'auth':
            return False
        password = self.password_of[user]
        A1 = '{}:{}:{}'.format(user, realm, password).encode('utf-8')
        A2 = '{}:{}'.format(command, auth['uri']).encode('utf-8')

        def H(x):
            return osutils.md5(x).hexdigest()

        def KD(secret, data):
            return H('{}:{}'.format(secret, data).encode('utf-8'))
        nonce_count = int(auth['nc'], 16)
        ncvalue = '%08x' % nonce_count
        cnonce = auth['cnonce']
        noncebit = '{}:{}:{}:{}:{}'.format(nonce, ncvalue, cnonce, qop, H(A2))
        response_digest = KD(H(A1), noncebit)
        return response_digest == auth['response']