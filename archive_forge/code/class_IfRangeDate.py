from webob.datetime_utils import (
from webob.descriptors import _rx_etag
from webob.util import header_docstring
class IfRangeDate(object):

    def __init__(self, date):
        self.date = date

    def __contains__(self, resp):
        last_modified = resp.last_modified
        return last_modified and last_modified <= self.date

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.date)

    def __str__(self):
        return serialize_date(self.date)