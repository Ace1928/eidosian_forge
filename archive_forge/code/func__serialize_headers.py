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
def _serialize_headers(headers: Iterable[bytes], request: Request) -> Generator[bytes, Any, None]:
    for header in headers:
        if header in request.headers:
            yield header
            for value in request.headers.getlist(header):
                yield value