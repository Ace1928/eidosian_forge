import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _normalized_cookie_tuples(self, attrs_set):
    """Return list of tuples containing normalised cookie information.

        attrs_set is the list of lists of key,value pairs extracted from
        the Set-Cookie or Set-Cookie2 headers.

        Tuples are name, value, standard, rest, where name and value are the
        cookie name and value, standard is a dictionary containing the standard
        cookie-attributes (discard, secure, version, expires or max-age,
        domain, path and port) and rest is a dictionary containing the rest of
        the cookie-attributes.

        """
    cookie_tuples = []
    boolean_attrs = ('discard', 'secure')
    value_attrs = ('version', 'expires', 'max-age', 'domain', 'path', 'port', 'comment', 'commenturl')
    for cookie_attrs in attrs_set:
        name, value = cookie_attrs[0]
        max_age_set = False
        bad_cookie = False
        standard = {}
        rest = {}
        for k, v in cookie_attrs[1:]:
            lc = k.lower()
            if lc in value_attrs or lc in boolean_attrs:
                k = lc
            if k in boolean_attrs and v is None:
                v = True
            if k in standard:
                continue
            if k == 'domain':
                if v is None:
                    _debug('   missing value for domain attribute')
                    bad_cookie = True
                    break
                v = v.lower()
            if k == 'expires':
                if max_age_set:
                    continue
                if v is None:
                    _debug('   missing or invalid value for expires attribute: treating as session cookie')
                    continue
            if k == 'max-age':
                max_age_set = True
                try:
                    v = int(v)
                except ValueError:
                    _debug('   missing or invalid (non-numeric) value for max-age attribute')
                    bad_cookie = True
                    break
                k = 'expires'
                v = self._now + v
            if k in value_attrs or k in boolean_attrs:
                if v is None and k not in ('port', 'comment', 'commenturl'):
                    _debug('   missing value for %s attribute' % k)
                    bad_cookie = True
                    break
                standard[k] = v
            else:
                rest[k] = v
        if bad_cookie:
            continue
        cookie_tuples.append((name, value, standard, rest))
    return cookie_tuples