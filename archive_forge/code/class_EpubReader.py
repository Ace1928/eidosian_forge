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
class EpubReader(object):
    DEFAULT_OPTIONS = {'ignore_ncx': False}

    def __init__(self, epub_file_name, options=None):
        self.file_name = epub_file_name
        self.book = EpubBook()
        self.zf = None
        self.opf_file = ''
        self.opf_dir = ''
        self.options = dict(self.DEFAULT_OPTIONS)
        if options:
            self.options.update(options)
        self._check_deprecated()

    def _check_deprecated(self):
        if not self.options.get('ignore_ncx'):
            warnings.warn('In the future version we will turn default option ignore_ncx to True.')

    def process(self):
        for plg in self.options.get('plugins', []):
            if hasattr(plg, 'after_read'):
                plg.after_read(self.book)
        for item in self.book.get_items():
            if isinstance(item, EpubHtml):
                for plg in self.options.get('plugins', []):
                    if hasattr(plg, 'html_after_read'):
                        plg.html_after_read(self.book, item)

    def load(self):
        self._load()
        return self.book

    def read_file(self, name):
        name = zip_path.normpath(name)
        return self.zf.read(name)

    def _load_container(self):
        meta_inf = self.read_file('META-INF/container.xml')
        tree = parse_string(meta_inf)
        for root_file in tree.findall('//xmlns:rootfile[@media-type]', namespaces={'xmlns': NAMESPACES['CONTAINERNS']}):
            if root_file.get('media-type') == 'application/oebps-package+xml':
                self.opf_file = root_file.get('full-path')
                self.opf_dir = zip_path.dirname(self.opf_file)

    def _load_metadata(self):
        container_root = self.container.getroot()
        self.book.version = container_root.get('version', None)
        if container_root.get('unique-identifier', None):
            self.book.IDENTIFIER_ID = container_root.get('unique-identifier')
        metadata = self.container.find('{%s}%s' % (NAMESPACES['OPF'], 'metadata'))
        nsmap = metadata.nsmap
        nstags = dict(((k, '{%s}' % v) for k, v in six.iteritems(nsmap)))
        default_ns = nstags.get(None, '')
        nsdict = dict(((v, {}) for v in nsmap.values()))

        def add_item(ns, tag, value, extra):
            if ns not in nsdict:
                nsdict[ns] = {}
            values = nsdict[ns].setdefault(tag, [])
            values.append((value, extra))
        for t in metadata:
            if not etree.iselement(t) or t.tag is etree.Comment:
                continue
            if t.tag == default_ns + 'meta':
                name = t.get('name')
                others = dict(((k, v) for k, v in t.items()))
                if name and ':' in name:
                    prefix, name = name.split(':', 1)
                else:
                    prefix = None
                add_item(t.nsmap.get(prefix, prefix), name, t.text, others)
            else:
                tag = t.tag[t.tag.rfind('}') + 1:]
                if (t.prefix and t.prefix.lower() == 'dc') and tag == 'identifier':
                    _id = t.get('id', None)
                    if _id:
                        self.book.IDENTIFIER_ID = _id
                others = dict(((k, v) for k, v in t.items()))
                add_item(t.nsmap[t.prefix], tag, t.text, others)
        self.book.metadata = nsdict
        titles = self.book.get_metadata('DC', 'title')
        if len(titles) > 0:
            self.book.title = titles[0][0]
        for value, others in self.book.get_metadata('DC', 'identifier'):
            if others.get('id') == self.book.IDENTIFIER_ID:
                self.book.uid = value

    def _load_manifest(self):
        for r in self.container.find('{%s}%s' % (NAMESPACES['OPF'], 'manifest')):
            if r is not None and r.tag != '{%s}item' % NAMESPACES['OPF']:
                continue
            media_type = r.get('media-type')
            _properties = r.get('properties', '')
            if _properties:
                properties = _properties.split(' ')
            else:
                properties = []
            if media_type == 'image/jpg':
                media_type = 'image/jpeg'
            if media_type == 'application/x-dtbncx+xml':
                ei = EpubNcx(uid=r.get('id'), file_name=unquote(r.get('href')))
                ei.content = self.read_file(zip_path.join(self.opf_dir, ei.file_name))
            elif media_type == 'application/smil+xml':
                ei = EpubSMIL(uid=r.get('id'), file_name=unquote(r.get('href')))
                ei.content = self.read_file(zip_path.join(self.opf_dir, ei.file_name))
            elif media_type == 'application/xhtml+xml':
                if 'nav' in properties:
                    ei = EpubNav(uid=r.get('id'), file_name=unquote(r.get('href')))
                    ei.content = self.read_file(zip_path.join(self.opf_dir, r.get('href')))
                elif 'cover' in properties:
                    ei = EpubCoverHtml()
                    ei.content = self.read_file(zip_path.join(self.opf_dir, unquote(r.get('href'))))
                else:
                    ei = EpubHtml()
                    ei.id = r.get('id')
                    ei.file_name = unquote(r.get('href'))
                    ei.media_type = media_type
                    ei.media_overlay = r.get('media-overlay', None)
                    ei.media_duration = r.get('duration', None)
                    ei.content = self.read_file(zip_path.join(self.opf_dir, ei.get_name()))
                    ei.properties = properties
            elif media_type in IMAGE_MEDIA_TYPES:
                if 'cover-image' in properties:
                    ei = EpubCover(uid=r.get('id'), file_name=unquote(r.get('href')))
                    ei.media_type = media_type
                    ei.content = self.read_file(zip_path.join(self.opf_dir, ei.get_name()))
                else:
                    ei = EpubImage()
                    ei.id = r.get('id')
                    ei.file_name = unquote(r.get('href'))
                    ei.media_type = media_type
                    ei.content = self.read_file(zip_path.join(self.opf_dir, ei.get_name()))
            else:
                ei = EpubItem()
                ei.id = r.get('id')
                ei.file_name = unquote(r.get('href'))
                ei.media_type = media_type
                ei.content = self.read_file(zip_path.join(self.opf_dir, ei.get_name()))
            self.book.add_item(ei)

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

    def _parse_nav(self, data, base_path, navtype='toc'):
        html_node = parse_html_string(data)
        if navtype == 'toc':
            nav_node = html_node.xpath("//nav[@*='toc']")[0]
        else:
            _page_list = html_node.xpath("//nav[@*='page-list']")
            if len(_page_list) == 0:
                return
            nav_node = _page_list[0]

        def parse_list(list_node):
            items = []
            for item_node in list_node.findall('li'):
                sublist_node = item_node.find('ol')
                link_node = item_node.find('a')
                if sublist_node is not None:
                    title = item_node[0].text
                    children = parse_list(sublist_node)
                    if link_node is not None:
                        href = zip_path.normpath(zip_path.join(base_path, link_node.get('href')))
                        items.append((Section(title, href=href), children))
                    else:
                        items.append((Section(title), children))
                elif link_node is not None:
                    title = link_node.text
                    href = zip_path.normpath(zip_path.join(base_path, link_node.get('href')))
                    items.append(Link(href, title))
            return items
        if navtype == 'toc':
            self.book.toc = parse_list(nav_node.find('ol'))
        elif nav_node is not None:
            self.book.pages = parse_list(nav_node.find('ol'))
            htmlfiles = dict()
            for htmlfile in self.book.items:
                if isinstance(htmlfile, EpubHtml):
                    htmlfiles[htmlfile.file_name] = htmlfile
            for page in self.book.pages:
                try:
                    filename, idref = page.href.split('#')
                except ValueError:
                    filename = page.href
                if filename in htmlfiles:
                    htmlfiles[filename].pages.append(page)

    def _load_spine(self):
        spine = self.container.find('{%s}%s' % (NAMESPACES['OPF'], 'spine'))
        self.book.spine = [(t.get('idref'), t.get('linear', 'yes')) for t in spine]
        toc = spine.get('toc', '')
        self.book.set_direction(spine.get('page-progression-direction', None))
        nav_item = next((item for item in self.book.items if isinstance(item, EpubNav)), None)
        if toc:
            if not self.options.get('ignore_ncx') or not nav_item:
                try:
                    ncxFile = self.read_file(zip_path.join(self.opf_dir, self.book.get_item_with_id(toc).get_name()))
                except KeyError:
                    raise EpubException(-1, 'Can not find ncx file.')
                self._parse_ncx(ncxFile)

    def _load_guide(self):
        guide = self.container.find('{%s}%s' % (NAMESPACES['OPF'], 'guide'))
        if guide is not None:
            self.book.guide = [{'href': t.get('href'), 'title': t.get('title'), 'type': t.get('type')} for t in guide]

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

    def _load(self):
        if os.path.isdir(self.file_name):
            file_name = self.file_name

            class Directory:

                def read(self, subname):
                    with open(os.path.join(file_name, subname), 'rb') as fp:
                        return fp.read()

                def close(self):
                    pass
            self.zf = Directory()
        else:
            try:
                self.zf = zipfile.ZipFile(self.file_name, 'r', compression=zipfile.ZIP_DEFLATED, allowZip64=True)
            except zipfile.BadZipfile as bz:
                raise EpubException(0, 'Bad Zip file')
            except zipfile.LargeZipFile as bz:
                raise EpubException(1, 'Large Zip file')
        self._load_container()
        self._load_opf_file()
        self.zf.close()