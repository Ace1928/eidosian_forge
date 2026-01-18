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
def _write_opf_bindings(self, root):
    if len(self.book.bindings) > 0:
        bindings = etree.SubElement(root, 'bindings', {})
        for item in self.book.bindings:
            etree.SubElement(bindings, 'mediaType', item)