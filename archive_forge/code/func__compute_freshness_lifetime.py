import gzip
import logging
import pickle
from email.utils import mktime_tz, parsedate_tz
from importlib import import_module
from pathlib import Path
from time import time
from weakref import WeakKeyDictionary
from w3lib.http import headers_dict_to_raw, headers_raw_to_dict
from scrapy.http import Headers, Response
from scrapy.http.request import Request
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.project import data_path
from scrapy.utils.python import to_bytes, to_unicode
def _compute_freshness_lifetime(self, response, request, now):
    cc = self._parse_cachecontrol(response)
    maxage = self._get_max_age(cc)
    if maxage is not None:
        return maxage
    date = rfc1123_to_epoch(response.headers.get(b'Date')) or now
    if b'Expires' in response.headers:
        expires = rfc1123_to_epoch(response.headers[b'Expires'])
        return max(0, expires - date) if expires else 0
    lastmodified = rfc1123_to_epoch(response.headers.get(b'Last-Modified'))
    if lastmodified and lastmodified <= date:
        return (date - lastmodified) / 10
    if response.status in (300, 301, 308):
        return self.MAXAGE
    return 0