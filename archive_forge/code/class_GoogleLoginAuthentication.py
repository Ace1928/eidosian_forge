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
class GoogleLoginAuthentication(Authentication):

    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        from urllib.parse import urlencode
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)
        challenge = auth._parse_www_authenticate(response, 'www-authenticate')
        service = challenge['googlelogin'].get('service', 'xapi')
        if service == 'xapi' and request_uri.find('calendar') > 0:
            service = 'cl'
        auth = dict(Email=credentials[0], Passwd=credentials[1], service=service, source=headers['user-agent'])
        resp, content = self.http.request('https://www.google.com/accounts/ClientLogin', method='POST', body=urlencode(auth), headers={'Content-Type': 'application/x-www-form-urlencoded'})
        lines = content.split('\n')
        d = dict([tuple(line.split('=', 1)) for line in lines if line])
        if resp.status == 403:
            self.Auth = ''
        else:
            self.Auth = d['Auth']

    def request(self, method, request_uri, headers, content):
        """Modify the request headers to add the appropriate
        Authorization header."""
        headers['authorization'] = 'GoogleLogin Auth=' + self.Auth