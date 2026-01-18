import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _set_cookie_strategy(self, strategy, have_lock=False):
    if strategy == self._cookie_strategy:
        return
    if not have_lock:
        self._cookie_lock.acquire()
    try:
        if self._cookie_strategy == 'csrf':
            utils.get_yf_logger().debug(f'toggling cookie strategy {self._cookie_strategy} -> basic')
            self._session.cookies.clear()
            self._cookie_strategy = 'basic'
        else:
            utils.get_yf_logger().debug(f'toggling cookie strategy {self._cookie_strategy} -> csrf')
            self._cookie_strategy = 'csrf'
        self._cookie = None
        self._crumb = None
    except Exception:
        self._cookie_lock.release()
        raise
    if not have_lock:
        self._cookie_lock.release()