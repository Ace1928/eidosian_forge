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
def _errno_from_exception(e):
    if len(e.args) > 0:
        return e.args[0].errno if isinstance(e.args[0], socket.error) else e.errno
    if hasattr(e, 'socket_err'):
        e_int = e.socket_err
        return e_int.args[0].errno if isinstance(e_int.args[0], socket.error) else e_int.errno
    return None