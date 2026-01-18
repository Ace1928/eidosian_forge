import base64
import datetime
import gzip
import io
import re
import struct
import urllib.parse
import urllib.request
import zlib
from .datetimes import _parse_date
from .urls import convert_to_idn
def _build_urllib2_request(url, agent, accept_header, etag, modified, referrer, auth, request_headers):
    request = urllib.request.Request(url)
    request.add_header('User-Agent', agent)
    if etag:
        request.add_header('If-None-Match', etag)
    if isinstance(modified, str):
        modified = _parse_date(modified)
    elif isinstance(modified, datetime.datetime):
        modified = modified.utctimetuple()
    if modified:
        short_weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        request.add_header('If-Modified-Since', '%s, %02d %s %04d %02d:%02d:%02d GMT' % (short_weekdays[modified[6]], modified[2], months[modified[1] - 1], modified[0], modified[3], modified[4], modified[5]))
    if referrer:
        request.add_header('Referer', referrer)
    request.add_header('Accept-encoding', 'gzip, deflate')
    if auth:
        request.add_header('Authorization', 'Basic %s' % auth)
    if accept_header:
        request.add_header('Accept', accept_header)
    for header_name, header_value in request_headers.items():
        request.add_header(header_name, header_value)
    request.add_header('A-IM', 'feed')
    return request