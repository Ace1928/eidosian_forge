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
def _updateCache(request_headers, response_headers, content, cache, cachekey):
    if cachekey:
        cc = _parse_cache_control(request_headers)
        cc_response = _parse_cache_control(response_headers)
        if 'no-store' in cc or 'no-store' in cc_response:
            cache.delete(cachekey)
        else:
            info = email.message.Message()
            for key, value in response_headers.items():
                if key not in ['status', 'content-encoding', 'transfer-encoding']:
                    info[key] = value
            vary = response_headers.get('vary', None)
            if vary:
                vary_headers = vary.lower().replace(' ', '').split(',')
                for header in vary_headers:
                    key = '-varied-%s' % header
                    try:
                        info[key] = request_headers[header]
                    except KeyError:
                        pass
            status = response_headers.status
            if status == 304:
                status = 200
            status_header = 'status: %d\r\n' % status
            try:
                header_str = info.as_string()
            except UnicodeEncodeError:
                setattr(info, '_write_headers', _bind_write_headers(info))
                header_str = info.as_string()
            header_str = re.sub('\r(?!\n)|(?<!\r)\n', '\r\n', header_str)
            text = b''.join([status_header.encode('utf-8'), header_str.encode('utf-8'), content])
            cache.set(cachekey, text)