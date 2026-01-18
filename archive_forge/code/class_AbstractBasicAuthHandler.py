import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
class AbstractBasicAuthHandler:
    rx = re.compile('(?:^|,)[ \t]*([^ \t,]+)[ \t]+realm=(["\']?)([^"\']*)\\2', re.I)

    def __init__(self, password_mgr=None):
        if password_mgr is None:
            password_mgr = HTTPPasswordMgr()
        self.passwd = password_mgr
        self.add_password = self.passwd.add_password

    def _parse_realm(self, header):
        found_challenge = False
        for mo in AbstractBasicAuthHandler.rx.finditer(header):
            scheme, quote, realm = mo.groups()
            if quote not in ['"', "'"]:
                warnings.warn('Basic Auth Realm was unquoted', UserWarning, 3)
            yield (scheme, realm)
            found_challenge = True
        if not found_challenge:
            if header:
                scheme = header.split()[0]
            else:
                scheme = ''
            yield (scheme, None)

    def http_error_auth_reqed(self, authreq, host, req, headers):
        headers = headers.get_all(authreq)
        if not headers:
            return
        unsupported = None
        for header in headers:
            for scheme, realm in self._parse_realm(header):
                if scheme.lower() != 'basic':
                    unsupported = scheme
                    continue
                if realm is not None:
                    return self.retry_http_basic_auth(host, req, realm)
        if unsupported is not None:
            raise ValueError('AbstractBasicAuthHandler does not support the following scheme: %r' % (scheme,))

    def retry_http_basic_auth(self, host, req, realm):
        user, pw = self.passwd.find_user_password(realm, host)
        if pw is not None:
            raw = '%s:%s' % (user, pw)
            auth = 'Basic ' + base64.b64encode(raw.encode()).decode('ascii')
            if req.get_header(self.auth_header, None) == auth:
                return None
            req.add_unredirected_header(self.auth_header, auth)
            return self.parent.open(req, timeout=req.timeout)
        else:
            return None

    def http_request(self, req):
        if not hasattr(self.passwd, 'is_authenticated') or not self.passwd.is_authenticated(req.full_url):
            return req
        if not req.has_header('Authorization'):
            user, passwd = self.passwd.find_user_password(None, req.full_url)
            credentials = '{0}:{1}'.format(user, passwd).encode()
            auth_str = base64.standard_b64encode(credentials).decode()
            req.add_unredirected_header('Authorization', 'Basic {}'.format(auth_str.strip()))
        return req

    def http_response(self, req, response):
        if hasattr(self.passwd, 'is_authenticated'):
            if 200 <= response.code < 300:
                self.passwd.update_authenticated(req.full_url, True)
            else:
                self.passwd.update_authenticated(req.full_url, False)
        return response
    https_request = http_request
    https_response = http_response