import datetime
import io
import json
import mimetypes
import os
import re
import sys
import time
import warnings
from email.header import Header
from http.client import responses
from urllib.parse import urlparse
from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.core import signals, signing
from django.core.exceptions import DisallowedRedirect
from django.core.serializers.json import DjangoJSONEncoder
from django.http.cookie import SimpleCookie
from django.utils import timezone
from django.utils.datastructures import CaseInsensitiveMapping
from django.utils.encoding import iri_to_uri
from django.utils.http import content_disposition_header, http_date
from django.utils.regex_helper import _lazy_re_compile
class HttpResponseBase:
    """
    An HTTP response base class with dictionary-accessed headers.

    This class doesn't handle content. It should not be used directly.
    Use the HttpResponse and StreamingHttpResponse subclasses instead.
    """
    status_code = 200

    def __init__(self, content_type=None, status=None, reason=None, charset=None, headers=None):
        self.headers = ResponseHeaders(headers)
        self._charset = charset
        if 'Content-Type' not in self.headers:
            if content_type is None:
                content_type = f'text/html; charset={self.charset}'
            self.headers['Content-Type'] = content_type
        elif content_type:
            raise ValueError("'headers' must not contain 'Content-Type' when the 'content_type' parameter is provided.")
        self._resource_closers = []
        self._handler_class = None
        self.cookies = SimpleCookie()
        self.closed = False
        if status is not None:
            try:
                self.status_code = int(status)
            except (ValueError, TypeError):
                raise TypeError('HTTP status code must be an integer.')
            if not 100 <= self.status_code <= 599:
                raise ValueError('HTTP status code must be an integer from 100 to 599.')
        self._reason_phrase = reason

    @property
    def reason_phrase(self):
        if self._reason_phrase is not None:
            return self._reason_phrase
        return responses.get(self.status_code, 'Unknown Status Code')

    @reason_phrase.setter
    def reason_phrase(self, value):
        self._reason_phrase = value

    @property
    def charset(self):
        if self._charset is not None:
            return self._charset
        if (content_type := self.headers.get('Content-Type')):
            if (matched := _charset_from_content_type_re.search(content_type)):
                return matched['charset'].replace('"', '')
        return settings.DEFAULT_CHARSET

    @charset.setter
    def charset(self, value):
        self._charset = value

    def serialize_headers(self):
        """HTTP headers as a bytestring."""
        return b'\r\n'.join([key.encode('ascii') + b': ' + value.encode('latin-1') for key, value in self.headers.items()])
    __bytes__ = serialize_headers

    @property
    def _content_type_for_repr(self):
        return ', "%s"' % self.headers['Content-Type'] if 'Content-Type' in self.headers else ''

    def __setitem__(self, header, value):
        self.headers[header] = value

    def __delitem__(self, header):
        del self.headers[header]

    def __getitem__(self, header):
        return self.headers[header]

    def has_header(self, header):
        """Case-insensitive check for a header."""
        return header in self.headers
    __contains__ = has_header

    def items(self):
        return self.headers.items()

    def get(self, header, alternate=None):
        return self.headers.get(header, alternate)

    def set_cookie(self, key, value='', max_age=None, expires=None, path='/', domain=None, secure=False, httponly=False, samesite=None):
        """
        Set a cookie.

        ``expires`` can be:
        - a string in the correct format,
        - a naive ``datetime.datetime`` object in UTC,
        - an aware ``datetime.datetime`` object in any time zone.
        If it is a ``datetime.datetime`` object then calculate ``max_age``.

        ``max_age`` can be:
        - int/float specifying seconds,
        - ``datetime.timedelta`` object.
        """
        self.cookies[key] = value
        if expires is not None:
            if isinstance(expires, datetime.datetime):
                if timezone.is_naive(expires):
                    expires = timezone.make_aware(expires, datetime.timezone.utc)
                delta = expires - datetime.datetime.now(tz=datetime.timezone.utc)
                delta += datetime.timedelta(seconds=1)
                expires = None
                if max_age is not None:
                    raise ValueError("'expires' and 'max_age' can't be used together.")
                max_age = max(0, delta.days * 86400 + delta.seconds)
            else:
                self.cookies[key]['expires'] = expires
        else:
            self.cookies[key]['expires'] = ''
        if max_age is not None:
            if isinstance(max_age, datetime.timedelta):
                max_age = max_age.total_seconds()
            self.cookies[key]['max-age'] = int(max_age)
            if not expires:
                self.cookies[key]['expires'] = http_date(time.time() + max_age)
        if path is not None:
            self.cookies[key]['path'] = path
        if domain is not None:
            self.cookies[key]['domain'] = domain
        if secure:
            self.cookies[key]['secure'] = True
        if httponly:
            self.cookies[key]['httponly'] = True
        if samesite:
            if samesite.lower() not in ('lax', 'none', 'strict'):
                raise ValueError('samesite must be "lax", "none", or "strict".')
            self.cookies[key]['samesite'] = samesite

    def setdefault(self, key, value):
        """Set a header unless it has already been set."""
        self.headers.setdefault(key, value)

    def set_signed_cookie(self, key, value, salt='', **kwargs):
        value = signing.get_cookie_signer(salt=key + salt).sign(value)
        return self.set_cookie(key, value, **kwargs)

    def delete_cookie(self, key, path='/', domain=None, samesite=None):
        secure = key.startswith(('__Secure-', '__Host-')) or (samesite and samesite.lower() == 'none')
        self.set_cookie(key, max_age=0, path=path, domain=domain, secure=secure, expires='Thu, 01 Jan 1970 00:00:00 GMT', samesite=samesite)

    def make_bytes(self, value):
        """Turn a value into a bytestring encoded in the output charset."""
        if isinstance(value, (bytes, memoryview)):
            return bytes(value)
        if isinstance(value, str):
            return bytes(value.encode(self.charset))
        return str(value).encode(self.charset)

    def close(self):
        for closer in self._resource_closers:
            try:
                closer()
            except Exception:
                pass
        self._resource_closers.clear()
        self.closed = True
        signals.request_finished.send(sender=self._handler_class)

    def write(self, content):
        raise OSError('This %s instance is not writable' % self.__class__.__name__)

    def flush(self):
        pass

    def tell(self):
        raise OSError('This %s instance cannot tell its position' % self.__class__.__name__)

    def readable(self):
        return False

    def seekable(self):
        return False

    def writable(self):
        return False

    def writelines(self, lines):
        raise OSError('This %s instance is not writable' % self.__class__.__name__)