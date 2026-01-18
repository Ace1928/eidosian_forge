import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _DateHeader(_SingleValueHeader):
    """
    handle date-based headers

    This extends the ``_SingleValueHeader`` object with specific
    treatment of time values:

    - It overrides ``compose`` to provide a sole keyword argument
      ``time`` which is an offset in seconds from the current time.

    - A ``time`` method is provided which parses the given value
      and returns the current time value.
    """

    def compose(self, time=None, delta=None):
        time = time or now()
        if delta:
            assert type(delta) == int
            time += delta
        return (formatdate(time),)

    def parse(self, *args, **kwargs):
        """ return the time value (in seconds since 1970) """
        value = self.__call__(*args, **kwargs)
        if value:
            try:
                return mktime_tz(parsedate_tz(value))
            except (TypeError, OverflowError):
                raise HTTPBadRequest('Received an ill-formed timestamp for %s: %s\r\n' % (self.name, value))