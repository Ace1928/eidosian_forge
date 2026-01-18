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
class AnonAuthHandler(AuthHandler, HmacKeys):
    """
    Implements Anonymous requests.
    """
    capability = ['anon']

    def __init__(self, host, config, provider):
        AuthHandler.__init__(self, host, config, provider)
        HmacKeys.__init__(self, host, config, provider, anon=True)

    def add_auth(self, http_request, **kwargs):
        pass