import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class DefaultHostMatches(Matcher):
    """Matches requests from host that is equal to application's default_host.
    Always returns no match if ``X-Real-Ip`` header is present.
    """

    def __init__(self, application: Any, host_pattern: Pattern) -> None:
        self.application = application
        self.host_pattern = host_pattern

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        if 'X-Real-Ip' not in request.headers:
            if self.host_pattern.match(self.application.default_host):
                return {}
        return None