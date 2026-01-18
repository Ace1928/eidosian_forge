import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _load_cookie_basic(self):
    cookie_dict = cache.get_cookie_cache().lookup('basic')
    if cookie_dict is None:
        return None
    if cookie_dict['age'] > datetime.timedelta(days=1):
        return None
    utils.get_yf_logger().debug('loaded persistent cookie')
    return cookie_dict['cookie']