from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def _extract_item(self, node, items_seen, base_url, itemids):
    itemid = self.get_docid(node, itemids)
    if self.nested:
        if itemid in items_seen:
            return
        items_seen.add(itemid)
    item = {}
    if not self.nested:
        item['iid'] = itemid
    types = node.get('itemtype', '').split()
    if types:
        if not self.strict and len(types) == 1:
            item['type'] = types[0]
        else:
            item['type'] = types
        nodeid = node.get('itemid')
        if nodeid:
            item['id'] = nodeid.strip()
    properties = collections.defaultdict(list)
    for name, value in self._extract_properties(node, items_seen=items_seen, base_url=base_url, itemids=itemids):
        properties[name].append(value)
    refs = node.get('itemref', '').split()
    if refs:
        for refid in refs:
            for name, value in self._extract_property_refs(node, refid, items_seen=items_seen, base_url=base_url, itemids=itemids):
                properties[name].append(value)
    props = []
    for name, values in properties.items():
        if not self.strict and len(values) == 1:
            props.append((name, values[0]))
        else:
            props.append((name, values))
    if props:
        item['properties'] = dict(props)
    else:
        item['value'] = self._extract_property_value(node, force=True, items_seen=items_seen, base_url=base_url, itemids=itemids)
    if self.add_text_content:
        textContent = self._extract_textContent(node)
        if textContent:
            item['textContent'] = textContent
    if self.add_html_node:
        item['htmlNode'] = node
    return item