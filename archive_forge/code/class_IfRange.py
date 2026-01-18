from __future__ import annotations
from .. import http
class IfRange:
    """Very simple object that represents the `If-Range` header in parsed
    form.  It will either have neither a etag or date or one of either but
    never both.

    .. versionadded:: 0.7
    """

    def __init__(self, etag=None, date=None):
        self.etag = etag
        self.date = date

    def to_header(self):
        """Converts the object back into an HTTP header."""
        if self.date is not None:
            return http.http_date(self.date)
        if self.etag is not None:
            return http.quote_etag(self.etag)
        return ''

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return f'<{type(self).__name__} {str(self)!r}>'