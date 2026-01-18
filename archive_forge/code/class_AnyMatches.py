import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class AnyMatches(Matcher):
    """Matches any request."""

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        return {}