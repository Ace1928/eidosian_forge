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
class TocTreeprocessor(Treeprocessor):
    """ Step through document and build TOC. """

    def __init__(self, md: Markdown, config: dict[str, Any]):
        super().__init__(md)
        self.marker: str = config['marker']
        self.title: str = config['title']
        self.base_level = int(config['baselevel']) - 1
        self.slugify = config['slugify']
        self.sep = config['separator']
        self.toc_class = config['toc_class']
        self.title_class: str = config['title_class']
        self.use_anchors: bool = parseBoolValue(config['anchorlink'])
        self.anchorlink_class: str = config['anchorlink_class']
        self.use_permalinks = parseBoolValue(config['permalink'], False)
        if self.use_permalinks is None:
            self.use_permalinks = config['permalink']
        self.permalink_class: str = config['permalink_class']
        self.permalink_title: str = config['permalink_title']
        self.permalink_leading: bool | None = parseBoolValue(config['permalink_leading'], False)
        self.header_rgx = re.compile('[Hh][123456]')
        if isinstance(config['toc_depth'], str) and '-' in config['toc_depth']:
            self.toc_top, self.toc_bottom = [int(x) for x in config['toc_depth'].split('-')]
        else:
            self.toc_top = 1
            self.toc_bottom = int(config['toc_depth'])

    def iterparent(self, node: etree.Element) -> Iterator[tuple[etree.Element, etree.Element]]:
        """ Iterator wrapper to get allowed parent and child all at once. """
        for child in node:
            if not self.header_rgx.match(child.tag) and child.tag not in ['pre', 'code']:
                yield (node, child)
                yield from self.iterparent(child)

    def replace_marker(self, root: etree.Element, elem: etree.Element) -> None:
        """ Replace marker with elem. """
        for p, c in self.iterparent(root):
            text = ''.join(c.itertext()).strip()
            if not text:
                continue
            if c.text and c.text.strip() == self.marker and (len(c) == 0):
                for i in range(len(p)):
                    if p[i] == c:
                        p[i] = elem
                        break

    def set_level(self, elem: etree.Element) -> None:
        """ Adjust header level according to base level. """
        level = int(elem.tag[-1]) + self.base_level
        if level > 6:
            level = 6
        elem.tag = 'h%d' % level

    def add_anchor(self, c: etree.Element, elem_id: str) -> None:
        anchor = etree.Element('a')
        anchor.text = c.text
        anchor.attrib['href'] = '#' + elem_id
        anchor.attrib['class'] = self.anchorlink_class
        c.text = ''
        for elem in c:
            anchor.append(elem)
        while len(c):
            c.remove(c[0])
        c.append(anchor)

    def add_permalink(self, c: etree.Element, elem_id: str) -> None:
        permalink = etree.Element('a')
        permalink.text = '%spara;' % AMP_SUBSTITUTE if self.use_permalinks is True else self.use_permalinks
        permalink.attrib['href'] = '#' + elem_id
        permalink.attrib['class'] = self.permalink_class
        if self.permalink_title:
            permalink.attrib['title'] = self.permalink_title
        if self.permalink_leading:
            permalink.tail = c.text
            c.text = ''
            c.insert(0, permalink)
        else:
            c.append(permalink)

    def build_toc_div(self, toc_list: list) -> etree.Element:
        """ Return a string div given a toc list. """
        div = etree.Element('div')
        div.attrib['class'] = self.toc_class
        if self.title:
            header = etree.SubElement(div, 'span')
            if self.title_class:
                header.attrib['class'] = self.title_class
            header.text = self.title

        def build_etree_ul(toc_list: list, parent: etree.Element) -> etree.Element:
            ul = etree.SubElement(parent, 'ul')
            for item in toc_list:
                li = etree.SubElement(ul, 'li')
                link = etree.SubElement(li, 'a')
                link.text = item.get('name', '')
                link.attrib['href'] = '#' + item.get('id', '')
                if item['children']:
                    build_etree_ul(item['children'], li)
            return ul
        build_etree_ul(toc_list, div)
        if 'prettify' in self.md.treeprocessors:
            self.md.treeprocessors['prettify'].run(div)
        return div

    def run(self, doc: etree.Element) -> None:
        used_ids = set()
        for el in doc.iter():
            if 'id' in el.attrib:
                used_ids.add(el.attrib['id'])
        toc_tokens = []
        for el in doc.iter():
            if isinstance(el.tag, str) and self.header_rgx.match(el.tag):
                self.set_level(el)
                innerhtml = render_inner_html(remove_fnrefs(el), self.md)
                name = strip_tags(innerhtml)
                if 'id' not in el.attrib:
                    el.attrib['id'] = unique(self.slugify(html.unescape(name), self.sep), used_ids)
                data_toc_label = ''
                if 'data-toc-label' in el.attrib:
                    data_toc_label = run_postprocessors(unescape(el.attrib['data-toc-label']), self.md)
                    name = escape_cdata(strip_tags(data_toc_label))
                    del el.attrib['data-toc-label']
                if int(el.tag[-1]) >= self.toc_top and int(el.tag[-1]) <= self.toc_bottom:
                    toc_tokens.append({'level': int(el.tag[-1]), 'id': el.attrib['id'], 'name': name, 'html': innerhtml, 'data-toc-label': data_toc_label})
                if self.use_anchors:
                    self.add_anchor(el, el.attrib['id'])
                if self.use_permalinks not in [False, None]:
                    self.add_permalink(el, el.attrib['id'])
        toc_tokens = nest_toc_tokens(toc_tokens)
        div = self.build_toc_div(toc_tokens)
        if self.marker:
            self.replace_marker(doc, div)
        toc = self.md.serializer(div)
        for pp in self.md.postprocessors:
            toc = pp.run(toc)
        self.md.toc_tokens = toc_tokens
        self.md.toc = toc