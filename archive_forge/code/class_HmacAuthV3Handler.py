import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
class HmacAuthV3Handler(AuthHandler, HmacKeys):
    """Implements the new Version 3 HMAC authorization used by Route53."""
    capability = ['hmac-v3', 'route53', 'ses']

    def __init__(self, host, config, provider):
        AuthHandler.__init__(self, host, config, provider)
        HmacKeys.__init__(self, host, config, provider)

    def add_auth(self, http_request, **kwargs):
        headers = http_request.headers
        if 'Date' not in headers:
            headers['Date'] = formatdate(usegmt=True)
        if self._provider.security_token:
            key = self._provider.security_token_header
            headers[key] = self._provider.security_token
        b64_hmac = self.sign_string(headers['Date'])
        s = 'AWS3-HTTPS AWSAccessKeyId=%s,' % self._provider.access_key
        s += 'Algorithm=%s,Signature=%s' % (self.algorithm(), b64_hmac)
        headers['X-Amzn-Authorization'] = s