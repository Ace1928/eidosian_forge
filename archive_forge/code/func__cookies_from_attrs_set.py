import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _cookies_from_attrs_set(self, attrs_set, request):
    cookie_tuples = self._normalized_cookie_tuples(attrs_set)
    cookies = []
    for tup in cookie_tuples:
        cookie = self._cookie_from_cookie_tuple(tup, request)
        if cookie:
            cookies.append(cookie)
    return cookies