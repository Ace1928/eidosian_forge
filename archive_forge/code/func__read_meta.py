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
def _read_meta(self, spider: Spider, request: Request):
    rpath = Path(self._get_request_path(spider, request))
    metapath = rpath / 'pickled_meta'
    if not metapath.exists():
        return
    mtime = metapath.stat().st_mtime
    if 0 < self.expiration_secs < time() - mtime:
        return
    with self._open(metapath, 'rb') as f:
        return pickle.load(f)