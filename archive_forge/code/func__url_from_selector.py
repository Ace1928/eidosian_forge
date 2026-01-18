from __future__ import annotations
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Generator, Optional, Tuple
from urllib.parse import urljoin
import parsel
from w3lib.encoding import (
from w3lib.html import strip_html5_whitespace
from scrapy.http import Request
from scrapy.http.response import Response
from scrapy.utils.python import memoizemethod_noargs, to_unicode
from scrapy.utils.response import get_base_url
def _url_from_selector(sel):
    if isinstance(sel.root, str):
        return strip_html5_whitespace(sel.root)
    if not hasattr(sel.root, 'tag'):
        raise _InvalidSelector(f'Unsupported selector: {sel}')
    if sel.root.tag not in ('a', 'link'):
        raise _InvalidSelector(f'Only <a> and <link> elements are supported; got <{sel.root.tag}>')
    href = sel.root.get('href')
    if href is None:
        raise _InvalidSelector(f'<{sel.root.tag}> element has no href attribute: {sel}')
    return strip_html5_whitespace(href)