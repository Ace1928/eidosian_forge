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
def _content_type_params__get(self):
    """
        A dictionary of all the parameters in the content type.

        (This is not a view, set to change, modifications of the dict will not
        be applied otherwise.)
        """
    params = self.headers.get('Content-Type', '')
    if ';' not in params:
        return {}
    params = params.split(';', 1)[1]
    result = {}
    for match in _PARAM_RE.finditer(params):
        result[match.group(1)] = match.group(2) or match.group(3) or ''
    return result