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
def _write_opf_spine(self, root, ncx_id):
    spine_attributes = {'toc': ncx_id or 'ncx'}
    if self.book.direction and self.options['spine_direction']:
        spine_attributes['page-progression-direction'] = self.book.direction
    spine = etree.SubElement(root, 'spine', spine_attributes)
    for _item in self.book.spine:
        is_linear = True
        if isinstance(_item, tuple):
            item = _item[0]
            if len(_item) > 1:
                if _item[1] == 'no':
                    is_linear = False
        else:
            item = _item
        if isinstance(item, EpubHtml):
            opts = {'idref': item.get_id()}
            if not item.is_linear or not is_linear:
                opts['linear'] = 'no'
        elif isinstance(item, EpubItem):
            opts = {'idref': item.get_id()}
            if not item.is_linear or not is_linear:
                opts['linear'] = 'no'
        else:
            opts = {'idref': item}
            try:
                itm = self.book.get_item_with_id(item)
                if not itm.is_linear or not is_linear:
                    opts['linear'] = 'no'
            except:
                pass
        etree.SubElement(spine, 'itemref', opts)