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
class ReferrerPolicy:
    NOREFERRER_SCHEMES: Tuple[str, ...] = LOCAL_SCHEMES
    name: str

    def referrer(self, response_url, request_url):
        raise NotImplementedError()

    def stripped_referrer(self, url):
        if urlparse(url).scheme not in self.NOREFERRER_SCHEMES:
            return self.strip_url(url)

    def origin_referrer(self, url):
        if urlparse(url).scheme not in self.NOREFERRER_SCHEMES:
            return self.origin(url)

    def strip_url(self, url, origin_only=False):
        """
        https://www.w3.org/TR/referrer-policy/#strip-url

        If url is null, return no referrer.
        If url's scheme is a local scheme, then return no referrer.
        Set url's username to the empty string.
        Set url's password to null.
        Set url's fragment to null.
        If the origin-only flag is true, then:
            Set url's path to null.
            Set url's query to null.
        Return url.
        """
        if not url:
            return None
        return strip_url(url, strip_credentials=True, strip_fragment=True, strip_default_port=True, origin_only=origin_only)

    def origin(self, url):
        """Return serialized origin (scheme, host, path) for a request or response URL."""
        return self.strip_url(url, origin_only=True)

    def potentially_trustworthy(self, url):
        parsed_url = urlparse(url)
        if parsed_url.scheme in ('data',):
            return False
        return self.tls_protected(url)

    def tls_protected(self, url):
        return urlparse(url).scheme in ('https', 'ftps')