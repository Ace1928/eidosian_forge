from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet
def _html_sub(m: re.Match[str]) -> str:
    """ Substitute raw html with plain text. """
    try:
        raw = md.htmlStash.rawHtmlBlocks[int(m.group(1))]
    except (IndexError, TypeError):
        return m.group(0)
    res = re.sub('(<[^>]+>)', '', raw)
    if strip_entities:
        res = re.sub('(&[\\#a-zA-Z0-9]+;)', '', res)
    return res