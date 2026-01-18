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
class UnimplementedDigestAuthOptionError(HttpLib2ErrorWithResponse):
    pass