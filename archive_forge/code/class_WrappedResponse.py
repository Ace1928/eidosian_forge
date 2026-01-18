import re
import time
from http.cookiejar import CookieJar as _CookieJar
from http.cookiejar import DefaultCookiePolicy
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
class WrappedResponse:

    def __init__(self, response):
        self.response = response

    def info(self):
        return self

    def get_all(self, name, default=None):
        return [to_unicode(v, errors='replace') for v in self.response.headers.getlist(name)]