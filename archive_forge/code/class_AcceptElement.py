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
class AcceptElement(HeaderElement):
    """An element (with parameters) from an Accept* header's element list.

    AcceptElement objects are comparable; the more-preferred object will be
    "less than" the less-preferred object. They are also therefore sortable;
    if you sort a list of AcceptElement objects, they will be listed in
    priority order; the most preferred value will be first. Yes, it should
    have been the other way around, but it's too late to fix now.
    """

    @classmethod
    def from_str(cls, elementstr):
        qvalue = None
        atoms = q_separator.split(elementstr, 1)
        media_range = atoms.pop(0).strip()
        if atoms:
            qvalue = HeaderElement.from_str(atoms[0].strip())
        media_type, params = cls.parse(media_range)
        if qvalue is not None:
            params['q'] = qvalue
        return cls(media_type, params)

    @property
    def qvalue(self):
        """The qvalue, or priority, of this value."""
        val = self.params.get('q', '1')
        if isinstance(val, HeaderElement):
            val = val.value
        try:
            return float(val)
        except ValueError as val_err:
            'Fail client requests with invalid quality value.\n\n            Ref: https://github.com/cherrypy/cherrypy/issues/1370\n            '
            raise cherrypy.HTTPError(400, 'Malformed HTTP header: `{}`'.format(str(self))) from val_err

    def __cmp__(self, other):
        diff = builtins.cmp(self.qvalue, other.qvalue)
        if diff == 0:
            diff = builtins.cmp(str(self), str(other))
        return diff

    def __lt__(self, other):
        if self.qvalue == other.qvalue:
            return str(self) < str(other)
        else:
            return self.qvalue < other.qvalue