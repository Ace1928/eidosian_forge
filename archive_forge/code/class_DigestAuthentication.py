import base64
import calendar
import copy
import email
import email.feedparser
from email import header
import email.message
import email.utils
import errno
from gettext import gettext as _
import gzip
from hashlib import md5 as _md5
from hashlib import sha1 as _sha
import hmac
import http.client
import io
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.parse
import zlib
from . import auth
from .error import *
from .iri2uri import iri2uri
from httplib2 import certs
class DigestAuthentication(Authentication):
    """Only do qop='auth' and MD5, since that
    is all Apache currently implements"""

    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)
        self.challenge = auth._parse_www_authenticate(response, 'www-authenticate')['digest']
        qop = self.challenge.get('qop', 'auth')
        self.challenge['qop'] = 'auth' in [x.strip() for x in qop.split()] and 'auth' or None
        if self.challenge['qop'] is None:
            raise UnimplementedDigestAuthOptionError(_('Unsupported value for qop: %s.' % qop))
        self.challenge['algorithm'] = self.challenge.get('algorithm', 'MD5').upper()
        if self.challenge['algorithm'] != 'MD5':
            raise UnimplementedDigestAuthOptionError(_('Unsupported value for algorithm: %s.' % self.challenge['algorithm']))
        self.A1 = ''.join([self.credentials[0], ':', self.challenge['realm'], ':', self.credentials[1]])
        self.challenge['nc'] = 1

    def request(self, method, request_uri, headers, content, cnonce=None):
        """Modify the request headers"""
        H = lambda x: _md5(x.encode('utf-8')).hexdigest()
        KD = lambda s, d: H('%s:%s' % (s, d))
        A2 = ''.join([method, ':', request_uri])
        self.challenge['cnonce'] = cnonce or _cnonce()
        request_digest = '"%s"' % KD(H(self.A1), '%s:%s:%s:%s:%s' % (self.challenge['nonce'], '%08x' % self.challenge['nc'], self.challenge['cnonce'], self.challenge['qop'], H(A2)))
        headers['authorization'] = 'Digest username="%s", realm="%s", nonce="%s", uri="%s", algorithm=%s, response=%s, qop=%s, nc=%08x, cnonce="%s"' % (self.credentials[0], self.challenge['realm'], self.challenge['nonce'], request_uri, self.challenge['algorithm'], request_digest, self.challenge['qop'], self.challenge['nc'], self.challenge['cnonce'])
        if self.challenge.get('opaque'):
            headers['authorization'] += ', opaque="%s"' % self.challenge['opaque']
        self.challenge['nc'] += 1

    def response(self, response, content):
        if 'authentication-info' not in response:
            challenge = auth._parse_www_authenticate(response, 'www-authenticate').get('digest', {})
            if 'true' == challenge.get('stale'):
                self.challenge['nonce'] = challenge['nonce']
                self.challenge['nc'] = 1
                return True
        else:
            updated_challenge = auth._parse_authentication_info(response, 'authentication-info')
            if 'nextnonce' in updated_challenge:
                self.challenge['nonce'] = updated_challenge['nextnonce']
                self.challenge['nc'] = 1
        return False