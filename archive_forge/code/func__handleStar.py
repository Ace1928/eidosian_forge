import copy
import os
import re
import zlib
from binascii import hexlify
from html import escape
from typing import List, Optional
from urllib.parse import quote as _quote
from zope.interface import implementer
from incremental import Version
from twisted import copyright
from twisted.internet import address, interfaces
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from twisted.logger import Logger
from twisted.python import components, failure, reflect
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.spread.pb import Copyable, ViewPoint
from twisted.web import http, iweb, resource, util
from twisted.web.error import UnsupportedMethod
from twisted.web.http import unquote
def _handleStar(self):
    """
        Handle receiving a request whose path is '*'.

        RFC 7231 defines an OPTIONS * request as being something that a client
        can send as a low-effort way to probe server capabilities or readiness.
        Rather than bother the user with this, we simply fast-path it back to
        an empty 200 OK. Any non-OPTIONS verb gets a 405 Method Not Allowed
        telling the client they can only use OPTIONS.
        """
    if self.method == b'OPTIONS':
        self.setResponseCode(http.OK)
    else:
        self.setResponseCode(http.NOT_ALLOWED)
        self.setHeader(b'Allow', b'OPTIONS')
    self.setHeader(b'Content-Length', b'0')
    self.finish()