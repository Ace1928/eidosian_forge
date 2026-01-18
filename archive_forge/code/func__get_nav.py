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
def _get_nav(self, item):
    nav_xml = parse_string(self.book.get_template('nav'))
    root = nav_xml.getroot()
    root.set('lang', self.book.language)
    root.attrib['{%s}lang' % NAMESPACES['XML']] = self.book.language
    nav_dir_name = os.path.dirname(item.file_name)
    head = etree.SubElement(root, 'head')
    title = etree.SubElement(head, 'title')
    title.text = item.title or self.book.title
    for _link in item.links:
        _lnk = etree.SubElement(head, 'link', {'href': _link.get('href', ''), 'rel': 'stylesheet', 'type': 'text/css'})
    body = etree.SubElement(root, 'body')
    nav = etree.SubElement(body, 'nav', {'{%s}type' % NAMESPACES['EPUB']: 'toc', 'id': 'id', 'role': 'doc-toc'})
    content_title = etree.SubElement(nav, 'h2')
    content_title.text = item.title or self.book.title

    def _create_section(itm, items):
        ol = etree.SubElement(itm, 'ol')
        for item in items:
            if isinstance(item, tuple) or isinstance(item, list):
                li = etree.SubElement(ol, 'li')
                if isinstance(item[0], EpubHtml):
                    a = etree.SubElement(li, 'a', {'href': os.path.relpath(item[0].file_name, nav_dir_name)})
                elif isinstance(item[0], Section) and item[0].href != '':
                    a = etree.SubElement(li, 'a', {'href': os.path.relpath(item[0].href, nav_dir_name)})
                elif isinstance(item[0], Link):
                    a = etree.SubElement(li, 'a', {'href': os.path.relpath(item[0].href, nav_dir_name)})
                else:
                    a = etree.SubElement(li, 'span')
                a.text = item[0].title
                _create_section(li, item[1])
            elif isinstance(item, Link):
                li = etree.SubElement(ol, 'li')
                a = etree.SubElement(li, 'a', {'href': os.path.relpath(item.href, nav_dir_name)})
                a.text = item.title
            elif isinstance(item, EpubHtml):
                li = etree.SubElement(ol, 'li')
                a = etree.SubElement(li, 'a', {'href': os.path.relpath(item.file_name, nav_dir_name)})
                a.text = item.title
    _create_section(nav, self.book.toc)
    if len(self.book.guide) > 0 and self.options.get('epub3_landmark'):
        guide_to_landscape_map = {'notes': 'rearnotes', 'text': 'bodymatter'}
        guide_nav = etree.SubElement(body, 'nav', {'{%s}type' % NAMESPACES['EPUB']: 'landmarks'})
        guide_content_title = etree.SubElement(guide_nav, 'h2')
        guide_content_title.text = self.options.get('landmark_title', 'Guide')
        guild_ol = etree.SubElement(guide_nav, 'ol')
        for elem in self.book.guide:
            li_item = etree.SubElement(guild_ol, 'li')
            if 'item' in elem:
                chap = elem.get('item', None)
                if chap:
                    _href = chap.file_name
                    _title = chap.title
            else:
                _href = elem.get('href', '')
                _title = elem.get('title', '')
            guide_type = elem.get('type', '')
            a_item = etree.SubElement(li_item, 'a', {'{%s}type' % NAMESPACES['EPUB']: guide_to_landscape_map.get(guide_type, guide_type), 'href': os.path.relpath(_href, nav_dir_name)})
            a_item.text = _title
    if self.options.get('epub3_pages'):
        inserted_pages = get_pages_for_items([item for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT) if not isinstance(item, EpubNav)])
        if len(inserted_pages) > 0:
            pagelist_nav = etree.SubElement(body, 'nav', {'{%s}type' % NAMESPACES['EPUB']: 'page-list', 'id': 'pages', 'hidden': 'hidden'})
            pagelist_content_title = etree.SubElement(pagelist_nav, 'h2')
            pagelist_content_title.text = self.options.get('pages_title', 'Pages')
            pages_ol = etree.SubElement(pagelist_nav, 'ol')
            for filename, pageref, label in inserted_pages:
                li_item = etree.SubElement(pages_ol, 'li')
                _href = u'{}#{}'.format(filename, pageref)
                _title = label
                a_item = etree.SubElement(li_item, 'a', {'href': os.path.relpath(_href, nav_dir_name)})
                a_item.text = _title
    tree_str = etree.tostring(nav_xml, pretty_print=True, encoding='utf-8', xml_declaration=True)
    return tree_str