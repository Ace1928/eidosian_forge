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
def _load_opf_file(self):
    try:
        s = self.read_file(self.opf_file)
    except KeyError:
        raise EpubException(-1, 'Can not find container file')
    self.container = parse_string(s)
    self._load_metadata()
    self._load_manifest()
    self._load_spine()
    self._load_guide()
    nav_item = next((item for item in self.book.items if isinstance(item, EpubNav)), None)
    if nav_item:
        if self.options.get('ignore_ncx') or not self.book.toc:
            self._parse_nav(nav_item.content, zip_path.dirname(nav_item.file_name), navtype='toc')
        self._parse_nav(nav_item.content, zip_path.dirname(nav_item.file_name), navtype='pages')