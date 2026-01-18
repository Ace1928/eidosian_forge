import logging
from collections import defaultdict
from tldextract import TLDExtract
from scrapy.exceptions import NotConfigured
from scrapy.http import Response
from scrapy.http.cookies import CookieJar
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def _get_request_cookies(self, jar, request):
    """
        Extract cookies from the Request.cookies attribute
        """
    if not request.cookies:
        return []
    if isinstance(request.cookies, dict):
        cookies = ({'name': k, 'value': v} for k, v in request.cookies.items())
    else:
        cookies = request.cookies
    formatted = filter(None, (self._format_cookie(c, request) for c in cookies))
    response = Response(request.url, headers={'Set-Cookie': formatted})
    return jar.make_cookies(response, request)