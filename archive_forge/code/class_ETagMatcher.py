from webob.datetime_utils import (
from webob.descriptors import _rx_etag
from webob.util import header_docstring
class ETagMatcher(object):

    def __init__(self, etags):
        self.etags = etags

    def __contains__(self, other):
        return other in self.etags

    def __repr__(self):
        return '<ETag %s>' % ' or '.join(self.etags)

    @classmethod
    def parse(cls, value, strong=True):
        """
        Parse this from a header value
        """
        if value == '*':
            return AnyETag
        if not value:
            return cls([])
        matches = _rx_etag.findall(value)
        if not matches:
            return cls([value])
        elif strong:
            return cls([t for w, t in matches if not w])
        else:
            return cls([t for w, t in matches])

    def __str__(self):
        return ', '.join(map('"%s"'.__mod__, self.etags))