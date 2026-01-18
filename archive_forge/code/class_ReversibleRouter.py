import re
from functools import partial
from tornado import httputil
from tornado.httpserver import _CallableAdapter
from tornado.escape import url_escape, url_unescape, utf8
from tornado.log import app_log
from tornado.util import basestring_type, import_object, re_unescape, unicode_type
from typing import Any, Union, Optional, Awaitable, List, Dict, Pattern, Tuple, overload
class ReversibleRouter(Router):
    """Abstract router interface for routers that can handle named routes
    and support reversing them to original urls.
    """

    def reverse_url(self, name: str, *args: Any) -> Optional[str]:
        """Returns url string for a given route name and arguments
        or ``None`` if no match is found.

        :arg str name: route name.
        :arg args: url parameters.
        :returns: parametrized url string for a given route name (or ``None``).
        """
        raise NotImplementedError()