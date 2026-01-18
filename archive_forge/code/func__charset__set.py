import re
import struct
import zlib
from base64 import b64encode
from datetime import datetime, timedelta
from hashlib import md5
from webob.byterange import ContentRange
from webob.cachecontrol import CacheControl, serialize_cache_control
from webob.compat import (
from webob.cookies import Cookie, make_cookie
from webob.datetime_utils import (
from webob.descriptors import (
from webob.headers import ResponseHeaders
from webob.request import BaseRequest
from webob.util import status_generic_reasons, status_reasons, warn_deprecation
def _charset__set(self, charset):
    if charset is None:
        self._charset__del()
        return
    header = self.headers.get('Content-Type', None)
    if header is None:
        raise AttributeError('You cannot set the charset when no content-type is defined')
    match = CHARSET_RE.search(header)
    if match:
        header = header[:match.start()] + header[match.end():]
    header += '; charset=%s' % charset
    self.headers['Content-Type'] = header