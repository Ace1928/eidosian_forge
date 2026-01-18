from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def _extract_property_value(self, node, items_seen, base_url, itemids, force=False):
    if not force and node.get('itemscope') is not None:
        if self.nested:
            return self._extract_item(node, items_seen=items_seen, base_url=base_url, itemids=itemids)
        else:
            return {'iid_ref': self.get_docid(node, itemids)}
    elif node.tag == 'meta':
        return node.get('content', '')
    elif node.tag in ('audio', 'embed', 'iframe', 'img', 'source', 'track', 'video'):
        return urljoin(base_url, strip_html5_whitespace(node.get('src', '')))
    elif node.tag in ('a', 'area', 'link'):
        return urljoin(base_url, strip_html5_whitespace(node.get('href', '')))
    elif node.tag in ('object',):
        return urljoin(base_url, strip_html5_whitespace(node.get('data', '')))
    elif node.tag in ('data', 'meter'):
        return node.get('value', '')
    elif node.tag in ('time',):
        return node.get('datetime', '')
    elif node.get('content'):
        return node.get('content')
    else:
        return self._extract_textContent(node)