from __future__ import annotations
import calendar
import time
from datetime import datetime, timedelta, timezone
from email.utils import formatdate, parsedate, parsedate_tz
from typing import TYPE_CHECKING, Any, Mapping
class LastModified(BaseHeuristic):
    """
    If there is no Expires header already, fall back on Last-Modified
    using the heuristic from
    http://tools.ietf.org/html/rfc7234#section-4.2.2
    to calculate a reasonable value.

    Firefox also does something like this per
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching_FAQ
    http://lxr.mozilla.org/mozilla-release/source/netwerk/protocol/http/nsHttpResponseHead.cpp#397
    Unlike mozilla we limit this to 24-hr.
    """
    cacheable_by_default_statuses = {200, 203, 204, 206, 300, 301, 404, 405, 410, 414, 501}

    def update_headers(self, resp: HTTPResponse) -> dict[str, str]:
        headers: Mapping[str, str] = resp.headers
        if 'expires' in headers:
            return {}
        if 'cache-control' in headers and headers['cache-control'] != 'public':
            return {}
        if resp.status not in self.cacheable_by_default_statuses:
            return {}
        if 'date' not in headers or 'last-modified' not in headers:
            return {}
        time_tuple = parsedate_tz(headers['date'])
        assert time_tuple is not None
        date = calendar.timegm(time_tuple[:6])
        last_modified = parsedate(headers['last-modified'])
        if last_modified is None:
            return {}
        now = time.time()
        current_age = max(0, now - date)
        delta = date - calendar.timegm(last_modified)
        freshness_lifetime = max(0, min(delta / 10, 24 * 3600))
        if freshness_lifetime <= current_age:
            return {}
        expires = date + freshness_lifetime
        return {'expires': time.strftime(TIME_FMT, time.gmtime(expires))}

    def warning(self, resp: HTTPResponse) -> str | None:
        return None