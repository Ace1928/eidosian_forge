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
def _app_iter__get(self):
    """
        Returns the ``app_iter`` of the response.

        If ``body`` was set, this will create an ``app_iter`` from that
        ``body`` (a single-item list).
        """
    return self._app_iter