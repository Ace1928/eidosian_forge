import logging
from collections import defaultdict
from tldextract import TLDExtract
from scrapy.exceptions import NotConfigured
from scrapy.http import Response
from scrapy.http.cookies import CookieJar
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def _debug_set_cookie(self, response, spider):
    if self.debug:
        cl = [to_unicode(c, errors='replace') for c in response.headers.getlist('Set-Cookie')]
        if cl:
            cookies = '\n'.join((f'Set-Cookie: {c}\n' for c in cl))
            msg = f'Received cookies from: {response}\n{cookies}'
            logger.debug(msg, extra={'spider': spider})