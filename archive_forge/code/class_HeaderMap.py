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
class HeaderMap(CaseInsensitiveDict):
    """A dict subclass for HTTP request and response headers.

    Each key is changed on entry to str(key).title(). This allows headers
    to be case-insensitive and avoid duplicates.

    Values are header values (decoded according to :rfc:`2047` if necessary).
    """
    protocol = (1, 1)
    encodings = ['ISO-8859-1']
    use_rfc_2047 = True

    def elements(self, key):
        """Return a sorted list of HeaderElements for the given header."""
        return header_elements(self.transform_key(key), self.get(key))

    def values(self, key):
        """Return a sorted list of HeaderElement.value for the given header."""
        return [e.value for e in self.elements(key)]

    def output(self):
        """Transform self into a list of (name, value) tuples."""
        return list(self.encode_header_items(self.items()))

    @classmethod
    def encode_header_items(cls, header_items):
        """
        Prepare the sequence of name, value tuples into a form suitable for
        transmitting on the wire for HTTP.
        """
        for k, v in header_items:
            if not isinstance(v, str) and (not isinstance(v, bytes)):
                v = str(v)
            yield tuple(map(cls.encode_header_item, (k, v)))

    @classmethod
    def encode_header_item(cls, item):
        if isinstance(item, str):
            item = cls.encode(item)
        return item.translate(header_translate_table, header_translate_deletechars)

    @classmethod
    def encode(cls, v):
        """Return the given header name or value, encoded for HTTP output."""
        for enc in cls.encodings:
            try:
                return v.encode(enc)
            except UnicodeEncodeError:
                continue
        if cls.protocol == (1, 1) and cls.use_rfc_2047:
            v = b2a_base64(v.encode('utf-8'))
            return b'=?utf-8?b?' + v.strip(b'\n') + b'?='
        raise ValueError('Could not encode header part %r using any of the encodings %r.' % (v, cls.encodings))