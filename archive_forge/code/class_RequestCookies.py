import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
class RequestCookies(MutableMapping):
    _cache_key = 'webob._parsed_cookies'

    def __init__(self, environ):
        self._environ = environ

    @property
    def _cache(self):
        env = self._environ
        header = env.get('HTTP_COOKIE', '')
        cache, cache_header = env.get(self._cache_key, ({}, None))
        if cache_header == header:
            return cache
        d = lambda b: b.decode('utf8')
        cache = dict(((d(k), d(v)) for k, v in parse_cookie(header)))
        env[self._cache_key] = (cache, header)
        return cache

    def _mutate_header(self, name, value):
        header = self._environ.get('HTTP_COOKIE')
        had_header = header is not None
        header = header or ''
        if not PY2:
            header = header.encode('latin-1')
        bytes_name = bytes_(name, 'ascii')
        if value is None:
            replacement = None
        else:
            bytes_val = _value_quote(bytes_(value, 'utf-8'))
            replacement = bytes_name + b'=' + bytes_val
        matches = _rx_cookie.finditer(header)
        found = False
        for match in matches:
            start, end = match.span()
            match_name = match.group(1)
            if match_name == bytes_name:
                found = True
                if replacement is None:
                    header = header[:start].rstrip(b' ;') + header[end:]
                else:
                    header = header[:start] + replacement + header[end:]
                break
        else:
            if replacement is not None:
                if header:
                    header += b'; ' + replacement
                else:
                    header = replacement
        if header:
            self._environ['HTTP_COOKIE'] = native_(header, 'latin-1')
        elif had_header:
            self._environ['HTTP_COOKIE'] = ''
        return found

    def _valid_cookie_name(self, name):
        if not isinstance(name, string_types):
            raise TypeError(name, 'cookie name must be a string')
        if not isinstance(name, text_type):
            name = text_(name, 'utf-8')
        try:
            bytes_cookie_name = bytes_(name, 'ascii')
        except UnicodeEncodeError:
            raise TypeError('cookie name must be encodable to ascii')
        if not _valid_cookie_name(bytes_cookie_name):
            raise TypeError('cookie name must be valid according to RFC 6265')
        return name

    def __setitem__(self, name, value):
        name = self._valid_cookie_name(name)
        if not isinstance(value, string_types):
            raise ValueError(value, 'cookie value must be a string')
        if not isinstance(value, text_type):
            try:
                value = text_(value, 'utf-8')
            except UnicodeDecodeError:
                raise ValueError(value, 'cookie value must be utf-8 binary or unicode')
        self._mutate_header(name, value)

    def __getitem__(self, name):
        return self._cache[name]

    def get(self, name, default=None):
        return self._cache.get(name, default)

    def __delitem__(self, name):
        name = self._valid_cookie_name(name)
        found = self._mutate_header(name, None)
        if not found:
            raise KeyError(name)

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()
    if PY2:

        def iterkeys(self):
            return self._cache.iterkeys()

        def itervalues(self):
            return self._cache.itervalues()

        def iteritems(self):
            return self._cache.iteritems()

    def __contains__(self, name):
        return name in self._cache

    def __iter__(self):
        return self._cache.__iter__()

    def __len__(self):
        return len(self._cache)

    def clear(self):
        self._environ['HTTP_COOKIE'] = ''

    def __repr__(self):
        return '<RequestCookies (dict-like) with values %r>' % (self._cache,)