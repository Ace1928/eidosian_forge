import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _get_cookie_basic(self, proxy=None, timeout=30):
    if self._cookie is not None:
        utils.get_yf_logger().debug('reusing cookie')
        return self._cookie
    self._cookie = self._load_cookie_basic()
    if self._cookie is not None:
        return self._cookie
    response = self._session.get(url='https://fc.yahoo.com', headers=self.user_agent_headers, proxies=proxy, timeout=timeout, allow_redirects=True)
    if not response.cookies:
        utils.get_yf_logger().debug('response.cookies = None')
        return None
    self._cookie = list(response.cookies)[0]
    if self._cookie == '':
        utils.get_yf_logger().debug("list(response.cookies)[0] = ''")
        return None
    self._save_cookie_basic(self._cookie)
    utils.get_yf_logger().debug(f'fetched basic cookie = {self._cookie}')
    return self._cookie