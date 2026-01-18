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
def _GetValidHostsForCert(self, cert):
    """Returns a list of valid host globs for an SSL certificate.

        Args:
          cert: A dictionary representing an SSL certificate.
        Returns:
          list: A list of valid host globs.
        """
    if 'subjectAltName' in cert:
        return [x[1] for x in cert['subjectAltName'] if x[0].lower() == 'dns']
    else:
        return [x[0][1] for x in cert['subject'] if x[0][0].lower() == 'commonname']