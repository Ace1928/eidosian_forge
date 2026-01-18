import csv
import logging
import re
from io import StringIO
from typing import (
from warnings import warn
from lxml import etree
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http import Response, TextResponse
from scrapy.selector import Selector
from scrapy.utils.python import re_rsearch, to_unicode
class _StreamReader:

    def __init__(self, obj: Union[Response, str, bytes]):
        self._ptr: int = 0
        self._text: Union[str, bytes]
        if isinstance(obj, TextResponse):
            self._text, self.encoding = (obj.body, obj.encoding)
        elif isinstance(obj, Response):
            self._text, self.encoding = (obj.body, 'utf-8')
        else:
            self._text, self.encoding = (obj, 'utf-8')
        self._is_unicode: bool = isinstance(self._text, str)
        self._is_first_read: bool = True

    def read(self, n: int=65535) -> bytes:
        method: Callable[[int], bytes] = self._read_unicode if self._is_unicode else self._read_string
        result = method(n)
        if self._is_first_read:
            self._is_first_read = False
            result = result.lstrip()
        return result

    def _read_string(self, n: int=65535) -> bytes:
        s, e = (self._ptr, self._ptr + n)
        self._ptr = e
        return cast(bytes, self._text)[s:e]

    def _read_unicode(self, n: int=65535) -> bytes:
        s, e = (self._ptr, self._ptr + n)
        self._ptr = e
        return cast(str, self._text)[s:e].encode('utf-8')