import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def _kill_elements(self, doc, condition, iterate=None):
    bad = []
    for el in doc.iter(iterate):
        if condition(el):
            bad.append(el)
    for el in bad:
        el.drop_tree()