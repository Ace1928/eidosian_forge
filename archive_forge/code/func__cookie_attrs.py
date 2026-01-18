import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _cookie_attrs(self, cookies):
    """Return a list of cookie-attributes to be returned to server.

        like ['foo="bar"; $Path="/"', ...]

        The $Version attribute is also added when appropriate (currently only
        once per request).

        """
    cookies.sort(key=lambda a: len(a.path), reverse=True)
    version_set = False
    attrs = []
    for cookie in cookies:
        version = cookie.version
        if not version_set:
            version_set = True
            if version > 0:
                attrs.append('$Version=%s' % version)
        if cookie.value is not None and self.non_word_re.search(cookie.value) and (version > 0):
            value = self.quote_re.sub('\\\\\\1', cookie.value)
        else:
            value = cookie.value
        if cookie.value is None:
            attrs.append(cookie.name)
        else:
            attrs.append('%s=%s' % (cookie.name, value))
        if version > 0:
            if cookie.path_specified:
                attrs.append('$Path="%s"' % cookie.path)
            if cookie.domain.startswith('.'):
                domain = cookie.domain
                if not cookie.domain_initial_dot and domain.startswith('.'):
                    domain = domain[1:]
                attrs.append('$Domain="%s"' % domain)
            if cookie.port is not None:
                p = '$Port'
                if cookie.port_specified:
                    p = p + '="%s"' % cookie.port
                attrs.append(p)
    return attrs