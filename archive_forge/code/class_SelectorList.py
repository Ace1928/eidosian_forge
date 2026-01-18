from typing import Any, Optional, Type, Union
from parsel import Selector as _ParselSelector
from scrapy.http import HtmlResponse, TextResponse, XmlResponse
from scrapy.utils.python import to_bytes
from scrapy.utils.trackref import object_ref
class SelectorList(_ParselSelector.selectorlist_cls, object_ref):
    """
    The :class:`SelectorList` class is a subclass of the builtin ``list``
    class, which provides a few additional methods.
    """