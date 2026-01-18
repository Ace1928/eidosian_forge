import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class HostMatches(Matcher):
    """Matches requests from hosts specified by ``host_pattern`` regex."""

    def __init__(self, host_pattern: Union[str, Pattern]) -> None:
        if isinstance(host_pattern, basestring_type):
            if not host_pattern.endswith('$'):
                host_pattern += '$'
            self.host_pattern = re.compile(host_pattern)
        else:
            self.host_pattern = host_pattern

    def match(self, request: httputil.HTTPServerRequest) -> Optional[Dict[str, Any]]:
        if self.host_pattern.match(request.host_name):
            return {}
        return None