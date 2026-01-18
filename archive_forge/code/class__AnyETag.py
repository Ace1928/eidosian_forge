from webob.datetime_utils import (
from webob.descriptors import _rx_etag
from webob.util import header_docstring
class _AnyETag(object):
    """
    Represents an ETag of *, or a missing ETag when matching is 'safe'
    """

    def __repr__(self):
        return '<ETag *>'

    def __nonzero__(self):
        return False
    __bool__ = __nonzero__

    def __contains__(self, other):
        return True

    def __str__(self):
        return '*'