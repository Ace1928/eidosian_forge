import warnings
from typing import Tuple
from urllib.parse import urlparse
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_unicode
from scrapy.utils.url import strip_url
def _set_referer(self, r, response):
    if isinstance(r, Request):
        referrer = self.policy(response, r).referrer(response.url, r.url)
        if referrer is not None:
            r.headers.setdefault('Referer', referrer)
    return r