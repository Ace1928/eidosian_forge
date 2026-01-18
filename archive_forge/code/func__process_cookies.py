import logging
from collections import defaultdict
from tldextract import TLDExtract
from scrapy.exceptions import NotConfigured
from scrapy.http import Response
from scrapy.http.cookies import CookieJar
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def _process_cookies(self, cookies, *, jar, request):
    for cookie in cookies:
        cookie_domain = cookie.domain
        if cookie_domain.startswith('.'):
            cookie_domain = cookie_domain[1:]
        request_domain = urlparse_cached(request).hostname.lower()
        if cookie_domain and _is_public_domain(cookie_domain):
            if cookie_domain != request_domain:
                continue
            cookie.domain = request_domain
        jar.set_cookie_if_ok(cookie, request)