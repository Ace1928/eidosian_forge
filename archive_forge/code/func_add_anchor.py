import base64
import os
import re
import textwrap
import warnings
from urllib.parse import quote
from xml.etree.ElementTree import Element
import bleach
from defusedxml import ElementTree  # type:ignore[import-untyped]
from nbconvert.preprocessors.sanitize import _get_default_css_sanitizer
def add_anchor(html, anchor_link_text='¶'):
    """Add an id and an anchor-link to an html header

    For use on markdown headings
    """
    try:
        h = ElementTree.fromstring(html)
    except Exception:
        return html
    link = _convert_header_id(html2text(h))
    h.set('id', link)
    a = Element('a', {'class': 'anchor-link', 'href': '#' + link})
    try:
        a.append(ElementTree.fromstring(anchor_link_text))
    except Exception:
        a.text = anchor_link_text
    h.append(a)
    return ElementTree.tostring(h).decode(encoding='utf-8')