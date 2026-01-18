import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _cookies_for_domain(self, domain, request):
    cookies = []
    if not self._policy.domain_return_ok(domain, request):
        return []
    _debug('Checking %s for cookies to return', domain)
    cookies_by_path = self._cookies[domain]
    for path in cookies_by_path.keys():
        if not self._policy.path_return_ok(path, request):
            continue
        cookies_by_name = cookies_by_path[path]
        for cookie in cookies_by_name.values():
            if not self._policy.return_ok(cookie, request):
                _debug('   not returning cookie')
                continue
            _debug("   it's a match")
            cookies.append(cookie)
    return cookies