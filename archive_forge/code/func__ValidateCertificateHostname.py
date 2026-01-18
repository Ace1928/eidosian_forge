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
def _ValidateCertificateHostname(self, cert, hostname):
    """Validates that a given hostname is valid for an SSL certificate.

        Args:
          cert: A dictionary representing an SSL certificate.
          hostname: The hostname to test.
        Returns:
          bool: Whether or not the hostname is valid for this certificate.
        """
    hosts = self._GetValidHostsForCert(cert)
    for host in hosts:
        host_re = host.replace('.', '\\.').replace('*', '[^.]*')
        if re.search('^%s$' % (host_re,), hostname, re.I):
            return True
    return False