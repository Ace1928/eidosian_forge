import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
@utils.log_indent_decorator
def _get_cookie_and_crumb(self, proxy=None, timeout=30):
    cookie, crumb, strategy = (None, None, None)
    utils.get_yf_logger().debug(f"cookie_mode = '{self._cookie_strategy}'")
    with self._cookie_lock:
        if self._cookie_strategy == 'csrf':
            crumb = self._get_crumb_csrf()
            if crumb is None:
                self._set_cookie_strategy('basic', have_lock=True)
                cookie, crumb = self._get_cookie_and_crumb_basic(proxy, timeout)
        else:
            cookie, crumb = self._get_cookie_and_crumb_basic(proxy, timeout)
            if cookie is None or crumb is None:
                self._set_cookie_strategy('csrf', have_lock=True)
                crumb = self._get_crumb_csrf()
        strategy = self._cookie_strategy
    return (cookie, crumb, strategy)