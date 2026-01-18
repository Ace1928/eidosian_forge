from typing import Any, Optional, Type, Union
from parsel import Selector as _ParselSelector
from scrapy.http import HtmlResponse, TextResponse, XmlResponse
from scrapy.utils.python import to_bytes
from scrapy.utils.trackref import object_ref
def _response_from_text(text: Union[str, bytes], st: Optional[str]) -> TextResponse:
    rt: Type[TextResponse] = XmlResponse if st == 'xml' else HtmlResponse
    return rt(url='about:blank', encoding='utf-8', body=to_bytes(text, 'utf-8'))