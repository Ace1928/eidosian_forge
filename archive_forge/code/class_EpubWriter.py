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
class EpubWriter(object):
    DEFAULT_OPTIONS = {'epub2_guide': True, 'epub3_landmark': True, 'epub3_pages': True, 'landmark_title': 'Guide', 'pages_title': 'Pages', 'spine_direction': True, 'package_direction': False, 'play_order': {'enabled': False, 'start_from': 1}}

    def __init__(self, name, book, options=None):
        self.file_name = name
        self.book = book
        self.options = dict(self.DEFAULT_OPTIONS)
        if options:
            self.options.update(options)
        self._init_play_order()

    def _init_play_order(self):
        self._play_order = {'enabled': False, 'start_from': 1}
        try:
            self._play_order['enabled'] = self.options['play_order']['enabled']
            self._play_order['start_from'] = self.options['play_order']['start_from']
        except KeyError:
            pass

    def process(self):
        for plg in self.options.get('plugins', []):
            if hasattr(plg, 'before_write'):
                plg.before_write(self.book)
        for item in self.book.get_items():
            if isinstance(item, EpubHtml):
                for plg in self.options.get('plugins', []):
                    if hasattr(plg, 'html_before_write'):
                        plg.html_before_write(self.book, item)

    def _write_container(self):
        container_xml = CONTAINER_XML % {'folder_name': self.book.FOLDER_NAME}
        self.out.writestr(CONTAINER_PATH, container_xml)

    def _write_opf_metadata(self, root):
        nsmap = {'dc': NAMESPACES['DC'], 'opf': NAMESPACES['OPF']}
        nsmap.update(self.book.namespaces)
        metadata = etree.SubElement(root, 'metadata', nsmap=nsmap)
        el = etree.SubElement(metadata, 'meta', {'property': 'dcterms:modified'})
        if 'mtime' in self.options:
            mtime = self.options['mtime']
        else:
            import datetime
            mtime = datetime.datetime.now()
        el.text = mtime.strftime('%Y-%m-%dT%H:%M:%SZ')
        for ns_name, values in six.iteritems(self.book.metadata):
            if ns_name == NAMESPACES['OPF']:
                for values in values.values():
                    for v in values:
                        if 'property' in v[1] and v[1]['property'] == 'dcterms:modified':
                            continue
                        try:
                            el = etree.SubElement(metadata, 'meta', v[1])
                            if v[0]:
                                el.text = v[0]
                        except ValueError:
                            logging.error('Could not create metadata.')
            else:
                for name, values in six.iteritems(values):
                    for v in values:
                        try:
                            if ns_name:
                                el = etree.SubElement(metadata, '{%s}%s' % (ns_name, name), v[1])
                            else:
                                el = etree.SubElement(metadata, '%s' % name, v[1])
                            el.text = v[0]
                        except ValueError:
                            logging.error('Could not create metadata "{}".'.format(name))

    def _write_opf_manifest(self, root):
        manifest = etree.SubElement(root, 'manifest')
        _ncx_id = None
        for item in self.book.get_items():
            if not item.manifest:
                continue
            if isinstance(item, EpubNav):
                etree.SubElement(manifest, 'item', {'href': item.get_name(), 'id': item.id, 'media-type': item.media_type, 'properties': 'nav'})
            elif isinstance(item, EpubNcx):
                _ncx_id = item.id
                etree.SubElement(manifest, 'item', {'href': item.file_name, 'id': item.id, 'media-type': item.media_type})
            elif isinstance(item, EpubCover):
                etree.SubElement(manifest, 'item', {'href': item.file_name, 'id': item.id, 'media-type': item.media_type, 'properties': 'cover-image'})
            else:
                opts = {'href': item.file_name, 'id': item.id, 'media-type': item.media_type}
                if hasattr(item, 'properties') and len(item.properties) > 0:
                    opts['properties'] = ' '.join(item.properties)
                if hasattr(item, 'media_overlay') and item.media_overlay is not None:
                    opts['media-overlay'] = item.media_overlay
                if hasattr(item, 'media_duration') and item.media_duration is not None:
                    opts['duration'] = item.media_duration
                etree.SubElement(manifest, 'item', opts)
        return _ncx_id

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

    def _write_opf_guide(self, root):
        if len(self.book.guide) > 0 and self.options.get('epub2_guide'):
            guide = etree.SubElement(root, 'guide', {})
            for item in self.book.guide:
                if 'item' in item:
                    chap = item.get('item')
                    if chap:
                        _href = chap.file_name
                        _title = chap.title
                else:
                    _href = item.get('href', '')
                    _title = item.get('title', '')
                if _title is None:
                    _title = ''
                ref = etree.SubElement(guide, 'reference', {'type': item.get('type', ''), 'title': _title, 'href': _href})

    def _write_opf_bindings(self, root):
        if len(self.book.bindings) > 0:
            bindings = etree.SubElement(root, 'bindings', {})
            for item in self.book.bindings:
                etree.SubElement(bindings, 'mediaType', item)

    def _write_opf_file(self, root):
        tree_str = etree.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True)
        self.out.writestr('%s/content.opf' % self.book.FOLDER_NAME, tree_str)

    def _write_opf(self):
        package_attributes = {'xmlns': NAMESPACES['OPF'], 'unique-identifier': self.book.IDENTIFIER_ID, 'version': '3.0'}
        if self.book.direction and self.options['package_direction']:
            package_attributes['dir'] = self.book.direction
        root = etree.Element('package', package_attributes)
        prefixes = ['rendition: http://www.idpf.org/vocab/rendition/#'] + self.book.prefixes
        root.attrib['prefix'] = ' '.join(prefixes)
        self._write_opf_metadata(root)
        _ncx_id = self._write_opf_manifest(root)
        self._write_opf_spine(root, _ncx_id)
        self._write_opf_guide(root)
        self._write_opf_bindings(root)
        self._write_opf_file(root)

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

    def _write_items(self):
        for item in self.book.get_items():
            if isinstance(item, EpubNcx):
                self.out.writestr('%s/%s' % (self.book.FOLDER_NAME, item.file_name), self._get_ncx())
            elif isinstance(item, EpubNav):
                self.out.writestr('%s/%s' % (self.book.FOLDER_NAME, item.file_name), self._get_nav(item))
            elif item.manifest:
                self.out.writestr('%s/%s' % (self.book.FOLDER_NAME, item.file_name), item.get_content())
            else:
                self.out.writestr('%s' % item.file_name, item.get_content())

    def write(self):
        self.out = zipfile.ZipFile(self.file_name, 'w', zipfile.ZIP_DEFLATED)
        self.out.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        self._write_container()
        self._write_opf()
        self._write_items()
        self.out.close()