import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
class RemoteDisconnected(ConnectionResetError, BadStatusLine):

    def __init__(self, *pos, **kw):
        BadStatusLine.__init__(self, '')
        ConnectionResetError.__init__(self, *pos, **kw)