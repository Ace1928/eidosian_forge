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
def _build_ssl_context(disable_ssl_certificate_validation, ca_certs, cert_file=None, key_file=None, maximum_version=None, minimum_version=None, key_password=None):
    if not hasattr(ssl, 'SSLContext'):
        raise RuntimeError('httplib2 requires Python 3.2+ for ssl.SSLContext')
    context = ssl.SSLContext(DEFAULT_TLS_VERSION)
    context.verify_mode = ssl.CERT_NONE if disable_ssl_certificate_validation else ssl.CERT_REQUIRED
    if maximum_version is not None:
        if hasattr(context, 'maximum_version'):
            if isinstance(maximum_version, str):
                maximum_version = getattr(ssl.TLSVersion, maximum_version)
            context.maximum_version = maximum_version
        else:
            raise RuntimeError('setting tls_maximum_version requires Python 3.7 and OpenSSL 1.1 or newer')
    if minimum_version is not None:
        if hasattr(context, 'minimum_version'):
            if isinstance(minimum_version, str):
                minimum_version = getattr(ssl.TLSVersion, minimum_version)
            context.minimum_version = minimum_version
        else:
            raise RuntimeError('setting tls_minimum_version requires Python 3.7 and OpenSSL 1.1 or newer')
    if hasattr(context, 'check_hostname'):
        context.check_hostname = not disable_ssl_certificate_validation
    context.load_verify_locations(ca_certs)
    if cert_file:
        context.load_cert_chain(cert_file, key_file, key_password)
    return context