import functools
import email.utils
import re
import builtins
from binascii import b2a_base64
from cgi import parse_header
from email.header import decode_header
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote_plus
import jaraco.collections
import cherrypy
from cherrypy._cpcompat import ntob, ntou
class HeaderElement(object):
    """An element (with parameters) from an HTTP header's element list."""

    def __init__(self, value, params=None):
        self.value = value
        if params is None:
            params = {}
        self.params = params

    def __cmp__(self, other):
        return builtins.cmp(self.value, other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        p = [';%s=%s' % (k, v) for k, v in self.params.items()]
        return str('%s%s' % (self.value, ''.join(p)))

    def __bytes__(self):
        return ntob(self.__str__())

    def __unicode__(self):
        return ntou(self.__str__())

    @staticmethod
    def parse(elementstr):
        """Transform 'token;key=val' to ('token', {'key': 'val'})."""
        initial_value, params = parse_header(elementstr)
        return (initial_value, params)

    @classmethod
    def from_str(cls, elementstr):
        """Construct an instance from a string of the form 'token;key=val'."""
        ival, params = cls.parse(elementstr)
        return cls(ival, params)