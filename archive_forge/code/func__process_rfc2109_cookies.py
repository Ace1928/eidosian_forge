import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _process_rfc2109_cookies(self, cookies):
    rfc2109_as_ns = getattr(self._policy, 'rfc2109_as_netscape', None)
    if rfc2109_as_ns is None:
        rfc2109_as_ns = not self._policy.rfc2965
    for cookie in cookies:
        if cookie.version == 1:
            cookie.rfc2109 = True
            if rfc2109_as_ns:
                cookie.version = 0