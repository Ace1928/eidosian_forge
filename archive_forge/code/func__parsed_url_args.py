import re
from time import time
from typing import Optional, Tuple
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from twisted.internet import defer
from twisted.internet.protocol import ClientFactory
from twisted.web.http import HTTPClient
from scrapy import Request
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes, to_unicode
def _parsed_url_args(parsed: ParseResult) -> Tuple[bytes, bytes, bytes, int, bytes]:
    path_str = urlunparse(('', '', parsed.path or '/', parsed.params, parsed.query, ''))
    path = to_bytes(path_str, encoding='ascii')
    assert parsed.hostname is not None
    host = to_bytes(parsed.hostname, encoding='ascii')
    port = parsed.port
    scheme = to_bytes(parsed.scheme, encoding='ascii')
    netloc = to_bytes(parsed.netloc, encoding='ascii')
    if port is None:
        port = 443 if scheme == b'https' else 80
    return (scheme, netloc, host, port, path)