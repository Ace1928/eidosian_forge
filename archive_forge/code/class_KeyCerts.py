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
class KeyCerts(Credentials):
    """Identical to Credentials except that
    name/password are mapped to key/cert."""

    def add(self, key, cert, domain, password):
        self.credentials.append((domain.lower(), key, cert, password))

    def iter(self, domain):
        for cdomain, key, cert, password in self.credentials:
            if cdomain == '' or domain == cdomain:
                yield (key, cert, password)