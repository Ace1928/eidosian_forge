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
def _conn_request(self, conn, request_uri, method, body, headers):
    i = 0
    seen_bad_status_line = False
    while i < RETRIES:
        i += 1
        try:
            if conn.sock is None:
                conn.connect()
            conn.request(method, request_uri, body, headers)
        except socket.timeout:
            conn.close()
            raise
        except socket.gaierror:
            conn.close()
            raise ServerNotFoundError('Unable to find the server at %s' % conn.host)
        except socket.error as e:
            errno_ = _errno_from_exception(e)
            if errno_ in (errno.ENETUNREACH, errno.EADDRNOTAVAIL) and i < RETRIES:
                continue
            raise
        except http.client.HTTPException:
            if conn.sock is None:
                if i < RETRIES - 1:
                    conn.close()
                    conn.connect()
                    continue
                else:
                    conn.close()
                    raise
            if i < RETRIES - 1:
                conn.close()
                conn.connect()
                continue
            pass
        try:
            response = conn.getresponse()
        except (http.client.BadStatusLine, http.client.ResponseNotReady):
            if not seen_bad_status_line and i == 1:
                i = 0
                seen_bad_status_line = True
                conn.close()
                conn.connect()
                continue
            else:
                conn.close()
                raise
        except socket.timeout:
            raise
        except (socket.error, http.client.HTTPException):
            conn.close()
            if i == 0:
                conn.close()
                conn.connect()
                continue
            else:
                raise
        else:
            content = b''
            if method == 'HEAD':
                conn.close()
            else:
                content = response.read()
            response = Response(response)
            if method != 'HEAD':
                content = _decompressContent(response, content)
        break
    return (response, content)