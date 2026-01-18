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
def _get_ncx(self):
    ncx = parse_string(self.book.get_template('ncx'))
    root = ncx.getroot()
    head = etree.SubElement(root, 'head')
    uid = etree.SubElement(head, 'meta', {'content': self.book.uid, 'name': 'dtb:uid'})
    uid = etree.SubElement(head, 'meta', {'content': '0', 'name': 'dtb:depth'})
    uid = etree.SubElement(head, 'meta', {'content': '0', 'name': 'dtb:totalPageCount'})
    uid = etree.SubElement(head, 'meta', {'content': '0', 'name': 'dtb:maxPageNumber'})
    doc_title = etree.SubElement(root, 'docTitle')
    title = etree.SubElement(doc_title, 'text')
    title.text = self.book.title
    nav_map = etree.SubElement(root, 'navMap')

    def _add_play_order(nav_point):
        nav_point.set('playOrder', str(self._play_order['start_from']))
        self._play_order['start_from'] += 1

    def _create_section(itm, items, uid):
        for item in items:
            if isinstance(item, tuple) or isinstance(item, list):
                section, subsection = (item[0], item[1])
                np = etree.SubElement(itm, 'navPoint', {'id': section.get_id() if isinstance(section, EpubHtml) else 'sep_%d' % uid})
                if self._play_order['enabled']:
                    _add_play_order(np)
                nl = etree.SubElement(np, 'navLabel')
                nt = etree.SubElement(nl, 'text')
                nt.text = section.title
                href = ''
                if isinstance(section, EpubHtml):
                    href = section.file_name
                elif isinstance(section, Section) and section.href != '':
                    href = section.href
                elif isinstance(section, Link):
                    href = section.href
                nc = etree.SubElement(np, 'content', {'src': href})
                uid = _create_section(np, subsection, uid + 1)
            elif isinstance(item, Link):
                _parent = itm
                _content = _parent.find('content')
                if _content is not None:
                    if _content.get('src') == '':
                        _content.set('src', item.href)
                np = etree.SubElement(itm, 'navPoint', {'id': item.uid})
                if self._play_order['enabled']:
                    _add_play_order(np)
                nl = etree.SubElement(np, 'navLabel')
                nt = etree.SubElement(nl, 'text')
                nt.text = item.title
                nc = etree.SubElement(np, 'content', {'src': item.href})
            elif isinstance(item, EpubHtml):
                _parent = itm
                _content = _parent.find('content')
                if _content is not None:
                    if _content.get('src') == '':
                        _content.set('src', item.file_name)
                np = etree.SubElement(itm, 'navPoint', {'id': item.get_id()})
                if self._play_order['enabled']:
                    _add_play_order(np)
                nl = etree.SubElement(np, 'navLabel')
                nt = etree.SubElement(nl, 'text')
                nt.text = item.title
                nc = etree.SubElement(np, 'content', {'src': item.file_name})
        return uid
    _create_section(nav_map, self.book.toc, 0)
    tree_str = etree.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True)
    return tree_str