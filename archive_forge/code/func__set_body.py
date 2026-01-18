from typing import Generator, Tuple
from urllib.parse import urljoin
from scrapy.exceptions import NotSupported
from scrapy.http.common import obsolete_setter
from scrapy.http.headers import Headers
from scrapy.http.request import Request
from scrapy.link import Link
from scrapy.utils.trackref import object_ref
def _set_body(self, body):
    if body is None:
        self._body = b''
    elif not isinstance(body, bytes):
        raise TypeError('Response body must be bytes. If you want to pass unicode body use TextResponse or HtmlResponse.')
    else:
        self._body = body