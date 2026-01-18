import base64
from urllib.parse import unquote, urlunparse
from urllib.request import _parse_proxy, getproxies, proxy_bypass
from scrapy.exceptions import NotConfigured
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes
def _basic_auth_header(self, username, password):
    user_pass = to_bytes(f'{unquote(username)}:{unquote(password)}', encoding=self.auth_encoding)
    return base64.b64encode(user_pass)