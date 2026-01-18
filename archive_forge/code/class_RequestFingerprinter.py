import hashlib
import json
import warnings
from typing import (
from urllib.parse import urlunparse
from weakref import WeakKeyDictionary
from w3lib.http import basic_auth_header
from w3lib.url import canonicalize_url
from scrapy import Request, Spider
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_bytes, to_unicode
class RequestFingerprinter:
    """Default fingerprinter.

    It takes into account a canonical version
    (:func:`w3lib.url.canonicalize_url`) of :attr:`request.url
    <scrapy.http.Request.url>` and the values of :attr:`request.method
    <scrapy.http.Request.method>` and :attr:`request.body
    <scrapy.http.Request.body>`. It then generates an `SHA1
    <https://en.wikipedia.org/wiki/SHA-1>`_ hash.

    .. seealso:: :setting:`REQUEST_FINGERPRINTER_IMPLEMENTATION`.
    """

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    def __init__(self, crawler: Optional['Crawler']=None):
        if crawler:
            implementation = crawler.settings.get('REQUEST_FINGERPRINTER_IMPLEMENTATION')
        else:
            implementation = '2.6'
        if implementation == '2.6':
            message = "'2.6' is a deprecated value for the 'REQUEST_FINGERPRINTER_IMPLEMENTATION' setting.\n\nIt is also the default value. In other words, it is normal to get this warning if you have not defined a value for the 'REQUEST_FINGERPRINTER_IMPLEMENTATION' setting. This is so for backward compatibility reasons, but it will change in a future version of Scrapy.\n\nSee the documentation of the 'REQUEST_FINGERPRINTER_IMPLEMENTATION' setting for information on how to handle this deprecation."
            warnings.warn(message, category=ScrapyDeprecationWarning, stacklevel=2)
            self._fingerprint = _request_fingerprint_as_bytes
        elif implementation == '2.7':
            self._fingerprint = fingerprint
        else:
            raise ValueError(f"Got an invalid value on setting 'REQUEST_FINGERPRINTER_IMPLEMENTATION': {implementation!r}. Valid values are '2.6' (deprecated) and '2.7'.")

    def fingerprint(self, request: Request) -> bytes:
        return self._fingerprint(request)