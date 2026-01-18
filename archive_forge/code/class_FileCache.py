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
class FileCache(object):
    """Uses a local directory as a store for cached files.
    Not really safe to use if multiple threads or processes are going to
    be running on the same cache.
    """

    def __init__(self, cache, safe=safename):
        self.cache = cache
        self.safe = safe
        if not os.path.exists(cache):
            os.makedirs(self.cache)

    def get(self, key):
        retval = None
        cacheFullPath = os.path.join(self.cache, self.safe(key))
        try:
            f = open(cacheFullPath, 'rb')
            retval = f.read()
            f.close()
        except IOError:
            pass
        return retval

    def set(self, key, value):
        cacheFullPath = os.path.join(self.cache, self.safe(key))
        f = open(cacheFullPath, 'wb')
        f.write(value)
        f.close()

    def delete(self, key):
        cacheFullPath = os.path.join(self.cache, self.safe(key))
        if os.path.exists(cacheFullPath):
            os.remove(cacheFullPath)