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
def _compute_current_age(self, response, request, now):
    currentage = 0
    date = rfc1123_to_epoch(response.headers.get(b'Date')) or now
    if now > date:
        currentage = now - date
    if b'Age' in response.headers:
        try:
            age = int(response.headers[b'Age'])
            currentage = max(currentage, age)
        except ValueError:
            pass
    return currentage