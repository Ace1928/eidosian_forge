from typing import Any, Optional, Type, Union
from parsel import Selector as _ParselSelector
from scrapy.http import HtmlResponse, TextResponse, XmlResponse
from scrapy.utils.python import to_bytes
from scrapy.utils.trackref import object_ref
def _st(response: Optional[TextResponse], st: Optional[str]) -> str:
    if st is None:
        return 'xml' if isinstance(response, XmlResponse) else 'html'
    return st