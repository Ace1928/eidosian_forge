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
class DummyPolicy:

    def __init__(self, settings):
        self.ignore_schemes = settings.getlist('HTTPCACHE_IGNORE_SCHEMES')
        self.ignore_http_codes = [int(x) for x in settings.getlist('HTTPCACHE_IGNORE_HTTP_CODES')]

    def should_cache_request(self, request):
        return urlparse_cached(request).scheme not in self.ignore_schemes

    def should_cache_response(self, response, request):
        return response.status not in self.ignore_http_codes

    def is_cached_response_fresh(self, cachedresponse, request):
        return True

    def is_cached_response_valid(self, cachedresponse, response, request):
        return True