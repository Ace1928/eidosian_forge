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
def _auth_from_challenge(self, host, request_uri, headers, response, content):
    """A generator that creates Authorization objects
           that can be applied to requests.
        """
    challenges = auth._parse_www_authenticate(response, 'www-authenticate')
    for cred in self.credentials.iter(host):
        for scheme in AUTH_SCHEME_ORDER:
            if scheme in challenges:
                yield AUTH_SCHEME_CLASSES[scheme](cred, host, request_uri, headers, response, content, self)