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
def _charset__get(self):
    """
        Get/set the ``charset`` specified in ``Content-Type``.

        There is no checking to validate that a ``content_type`` actually
        allows for a ``charset`` parameter.
        """
    header = self.headers.get('Content-Type')
    if not header:
        return None
    match = CHARSET_RE.search(header)
    if match:
        return match.group(1)
    return None