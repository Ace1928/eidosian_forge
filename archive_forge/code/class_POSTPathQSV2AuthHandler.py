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
class POSTPathQSV2AuthHandler(QuerySignatureV2AuthHandler, AuthHandler):
    """
    Query Signature V2 Authentication relocating signed query
    into the path and allowing POST requests with Content-Types.
    """
    capability = ['mws']

    def add_auth(self, req, **kwargs):
        req.params['AWSAccessKeyId'] = self._provider.access_key
        req.params['SignatureVersion'] = self.SignatureVersion
        req.params['Timestamp'] = boto.utils.get_ts()
        qs, signature = self._calc_signature(req.params, req.method, req.auth_path, req.host)
        boto.log.debug('query_string: %s Signature: %s' % (qs, signature))
        if req.method == 'POST':
            req.headers['Content-Length'] = str(len(req.body))
            req.headers['Content-Type'] = req.headers.get('Content-Type', 'text/plain')
        else:
            req.body = ''
        req.path = req.path.split('?')[0]
        req.path = req.path + '?' + qs + '&Signature=' + urllib.parse.quote_plus(signature)