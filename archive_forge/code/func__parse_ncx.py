import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def _parse_ncx(self, data):
    tree = parse_string(data)
    tree_root = tree.getroot()
    nav_map = tree_root.find('{%s}navMap' % NAMESPACES['DAISY'])

    def _get_children(elems, n, nid):
        label, content = ('', '')
        children = []
        for a in elems.getchildren():
            if a.tag == '{%s}navLabel' % NAMESPACES['DAISY']:
                label = a.getchildren()[0].text
            if a.tag == '{%s}content' % NAMESPACES['DAISY']:
                content = a.get('src', '')
            if a.tag == '{%s}navPoint' % NAMESPACES['DAISY']:
                children.append(_get_children(a, n + 1, a.get('id', '')))
        if len(children) > 0:
            if n == 0:
                return children
            return (Section(label, href=content), children)
        else:
            return Link(content, label, nid)
    self.book.toc = _get_children(nav_map, 0, '')