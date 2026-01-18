import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
@utils.log_indent_decorator
def _get_crumb_csrf(self, proxy=None, timeout=30):
    if self._crumb is not None:
        utils.get_yf_logger().debug('reusing crumb')
        return self._crumb
    if not self._get_cookie_csrf(proxy, timeout):
        return None
    get_args = {'url': 'https://query2.finance.yahoo.com/v1/test/getcrumb', 'headers': self.user_agent_headers, 'proxies': proxy, 'timeout': timeout}
    if self._session_is_caching:
        get_args['expire_after'] = self._expire_after
        r = self._session.get(**get_args)
    else:
        r = self._session.get(**get_args)
    self._crumb = r.text
    if self._crumb is None or '<html>' in self._crumb or self._crumb == '':
        utils.get_yf_logger().debug("Didn't receive crumb")
        return None
    utils.get_yf_logger().debug(f"crumb = '{self._crumb}'")
    return self._crumb