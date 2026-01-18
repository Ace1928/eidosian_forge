from __future__ import print_function
import base64
import calendar
import copy
import email
import email.FeedParser
import email.Message
import email.Utils
import errno
import gzip
import httplib
import os
import random
import re
import StringIO
import sys
import time
import urllib
import urlparse
import zlib
import hmac
from gettext import gettext as _
import socket
from httplib2 import auth
from httplib2.error import *
from httplib2 import certs
class AppEngineHttpsConnection(httplib.HTTPSConnection):
    """Same as AppEngineHttpConnection, but for HTTPS URIs.

    The parameters proxy_info, ca_certs, disable_ssl_certificate_validation,
    and ssl_version are all dropped on the ground.
    """

    def __init__(self, host, port=None, key_file=None, cert_file=None, strict=None, timeout=None, proxy_info=None, ca_certs=None, disable_ssl_certificate_validation=False, ssl_version=None, key_password=None):
        if key_password:
            raise NotSupportedOnThisPlatform('Certificate with password is not supported.')
        httplib.HTTPSConnection.__init__(self, host, port=port, key_file=key_file, cert_file=cert_file, strict=strict, timeout=timeout)
        self._fetch = _new_fixed_fetch(not disable_ssl_certificate_validation)