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
class HmacDigestAuthentication(Authentication):
    """Adapted from Robert Sayre's code and DigestAuthentication above."""
    __author__ = 'Thomas Broyer (t.broyer@ltgt.net)'

    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)
        challenge = auth._parse_www_authenticate(response, 'www-authenticate')
        self.challenge = challenge['hmacdigest']
        self.challenge['reason'] = self.challenge.get('reason', 'unauthorized')
        if self.challenge['reason'] not in ['unauthorized', 'integrity']:
            self.challenge['reason'] = 'unauthorized'
        self.challenge['salt'] = self.challenge.get('salt', '')
        if not self.challenge.get('snonce'):
            raise UnimplementedHmacDigestAuthOptionError(_("The challenge doesn't contain a server nonce, or this one is empty."))
        self.challenge['algorithm'] = self.challenge.get('algorithm', 'HMAC-SHA-1')
        if self.challenge['algorithm'] not in ['HMAC-SHA-1', 'HMAC-MD5']:
            raise UnimplementedHmacDigestAuthOptionError(_('Unsupported value for algorithm: %s.' % self.challenge['algorithm']))
        self.challenge['pw-algorithm'] = self.challenge.get('pw-algorithm', 'SHA-1')
        if self.challenge['pw-algorithm'] not in ['SHA-1', 'MD5']:
            raise UnimplementedHmacDigestAuthOptionError(_('Unsupported value for pw-algorithm: %s.' % self.challenge['pw-algorithm']))
        if self.challenge['algorithm'] == 'HMAC-MD5':
            self.hashmod = _md5
        else:
            self.hashmod = _sha
        if self.challenge['pw-algorithm'] == 'MD5':
            self.pwhashmod = _md5
        else:
            self.pwhashmod = _sha
        self.key = ''.join([self.credentials[0], ':', self.pwhashmod.new(''.join([self.credentials[1], self.challenge['salt']])).hexdigest().lower(), ':', self.challenge['realm']])
        self.key = self.pwhashmod.new(self.key).hexdigest().lower()

    def request(self, method, request_uri, headers, content):
        """Modify the request headers"""
        keys = _get_end2end_headers(headers)
        keylist = ''.join(['%s ' % k for k in keys])
        headers_val = ''.join([headers[k] for k in keys])
        created = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        cnonce = _cnonce()
        request_digest = '%s:%s:%s:%s:%s' % (method, request_uri, cnonce, self.challenge['snonce'], headers_val)
        request_digest = hmac.new(self.key, request_digest, self.hashmod).hexdigest().lower()
        headers['authorization'] = 'HMACDigest username="%s", realm="%s", snonce="%s", cnonce="%s", uri="%s", created="%s", response="%s", headers="%s"' % (self.credentials[0], self.challenge['realm'], self.challenge['snonce'], cnonce, request_uri, created, request_digest, keylist)

    def response(self, response, content):
        challenge = auth._parse_www_authenticate(response, 'www-authenticate').get('hmacdigest', {})
        if challenge.get('reason') in ['integrity', 'stale']:
            return True
        return False