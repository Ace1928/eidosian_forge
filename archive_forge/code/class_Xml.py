import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
class Xml:

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __init__(self, rhs):
        if isinstance(rhs, mupdf.FzXml):
            self.this = rhs
        elif isinstance(str):
            buff = mupdf.fz_new_buffer_from_copied_data(rhs)
            self.this = mupdf.fz_parse_xml_from_html5(buff)
        else:
            assert 0, f'Unsupported type for rhs: {type(rhs)}'

    def _get_node_tree(self):

        def show_node(node, items, shift):
            while node is not None:
                if node.is_text:
                    items.append((shift, f'"{node.text}"'))
                    node = node.next
                    continue
                items.append((shift, f'({node.tagname}'))
                for k, v in node.get_attributes().items():
                    items.append((shift, f"={k} '{v}'"))
                child = node.first_child
                if child:
                    items = show_node(child, items, shift + 1)
                items.append((shift, f'){node.tagname}'))
                node = node.next
            return items
        shift = 0
        items = []
        items = show_node(self, items, shift)
        return items

    def add_bullet_list(self):
        """Add bulleted list ("ul" tag)"""
        child = self.create_element('ul')
        self.append_child(child)
        return child

    def add_class(self, text):
        """Set some class via CSS. Replaces complete class spec."""
        cls = self.get_attribute_value('class')
        if cls is not None and text in cls:
            return self
        self.remove_attribute('class')
        if cls is None:
            cls = text
        else:
            cls += ' ' + text
        self.set_attribute('class', cls)
        return self

    def add_code(self, text=None):
        """Add a "code" tag"""
        child = self.create_element('code')
        if type(text) is str:
            child.append_child(self.create_text_node(text))
        prev = self.span_bottom()
        if prev is None:
            prev = self
        prev.append_child(child)
        return self

    def add_codeblock(self):
        """Add monospaced lines ("pre" node)"""
        child = self.create_element('pre')
        self.append_child(child)
        return child

    def add_description_list(self):
        """Add description list ("dl" tag)"""
        child = self.create_element('dl')
        self.append_child(child)
        return child

    def add_division(self):
        """Add "div" tag"""
        child = self.create_element('div')
        self.append_child(child)
        return child

    def add_header(self, level=1):
        """Add header tag"""
        if level not in range(1, 7):
            raise ValueError('Header level must be in [1, 6]')
        this_tag = self.tagname
        new_tag = f'h{level}'
        child = self.create_element(new_tag)
        if this_tag not in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'):
            self.append_child(child)
            return child
        self.parent.append_child(child)
        return child

    def add_horizontal_line(self):
        """Add horizontal line ("hr" tag)"""
        child = self.create_element('hr')
        self.append_child(child)
        return child

    def add_image(self, name, width=None, height=None, imgfloat=None, align=None):
        """Add image node (tag "img")."""
        child = self.create_element('img')
        if width is not None:
            child.set_attribute('width', f'{width}')
        if height is not None:
            child.set_attribute('height', f'{height}')
        if imgfloat is not None:
            child.set_attribute('style', f'float: {imgfloat}')
        if align is not None:
            child.set_attribute('align', f'{align}')
        child.set_attribute('src', f'{name}')
        self.append_child(child)
        return child

    def add_link(self, href, text=None):
        """Add a hyperlink ("a" tag)"""
        child = self.create_element('a')
        if not isinstance(text, str):
            text = href
        child.set_attribute('href', href)
        child.append_child(self.create_text_node(text))
        prev = self.span_bottom()
        if prev is None:
            prev = self
        prev.append_child(child)
        return self

    def add_list_item(self):
        """Add item ("li" tag) under a (numbered or bulleted) list."""
        if self.tagname not in ('ol', 'ul'):
            raise ValueError('cannot add list item to', self.tagname)
        child = self.create_element('li')
        self.append_child(child)
        return child

    def add_number_list(self, start=1, numtype=None):
        """Add numbered list ("ol" tag)"""
        child = self.create_element('ol')
        if start > 1:
            child.set_attribute('start', str(start))
        if numtype is not None:
            child.set_attribute('type', numtype)
        self.append_child(child)
        return child

    def add_paragraph(self):
        """Add "p" tag"""
        child = self.create_element('p')
        if self.tagname != 'p':
            self.append_child(child)
        else:
            self.parent.append_child(child)
        return child

    def add_span(self):
        child = self.create_element('span')
        self.append_child(child)
        return child

    def add_style(self, text):
        """Set some style via CSS style. Replaces complete style spec."""
        style = self.get_attribute_value('style')
        if style is not None and text in style:
            return self
        self.remove_attribute('style')
        if style is None:
            style = text
        else:
            style += ';' + text
        self.set_attribute('style', style)
        return self

    def add_subscript(self, text=None):
        """Add a subscript ("sub" tag)"""
        child = self.create_element('sub')
        if type(text) is str:
            child.append_child(self.create_text_node(text))
        prev = self.span_bottom()
        if prev is None:
            prev = self
        prev.append_child(child)
        return self

    def add_superscript(self, text=None):
        """Add a superscript ("sup" tag)"""
        child = self.create_element('sup')
        if type(text) is str:
            child.append_child(self.create_text_node(text))
        prev = self.span_bottom()
        if prev is None:
            prev = self
        prev.append_child(child)
        return self

    def add_text(self, text):
        """Add text. Line breaks are honored."""
        lines = text.splitlines()
        line_count = len(lines)
        prev = self.span_bottom()
        if prev is None:
            prev = self
        for i, line in enumerate(lines):
            prev.append_child(self.create_text_node(line))
            if i < line_count - 1:
                prev.append_child(self.create_element('br'))
        return self

    def append_child(self, child):
        mupdf.fz_dom_append_child(self.this, child.this)

    def append_styled_span(self, style):
        span = self.create_element('span')
        span.add_style(style)
        prev = self.span_bottom()
        if prev is None:
            prev = self
        prev.append_child(span)
        return prev

    def bodytag(self):
        return Xml(mupdf.fz_dom_body(self.this))

    def clone(self):
        ret = mupdf.fz_dom_clone(self.this)
        return Xml(ret)

    @staticmethod
    def color_text(color):
        if type(color) is str:
            return color
        if type(color) is int:
            return f'rgb({sRGB_to_rgb(color)})'
        if type(color) in (tuple, list):
            return f'rgb{tuple(color)}'
        return color

    def create_element(self, tag):
        return Xml(mupdf.fz_dom_create_element(self.this, tag))

    def create_text_node(self, text):
        return Xml(mupdf.fz_dom_create_text_node(self.this, text))

    def debug(self):
        """Print a list of the node tree below self."""
        items = self._get_node_tree()
        for item in items:
            message('  ' * item[0] + item[1].replace('\n', '\\n'))

    def find(self, tag, att, match):
        ret = mupdf.fz_dom_find(self.this, tag, att, match)
        if ret.m_internal:
            return Xml(ret)

    def find_next(self, tag, att, match):
        ret = mupdf.fz_dom_find_next(self.this, tag, att, match)
        if ret.m_internal:
            return Xml(ret)

    @property
    def first_child(self):
        if mupdf.fz_xml_text(self.this):
            return
        ret = mupdf.fz_dom_first_child(self)
        if ret.m_internal:
            return Xml(ret)

    def get_attribute_value(self, key):
        assert key
        return mupdf.fz_dom_attribute(self.this, key)

    def get_attributes(self):
        if mupdf.fz_xml_text(self.this):
            return
        result = dict()
        i = 0
        while 1:
            val, key = mupdf.fz_dom_get_attribute(self.this, i)
            if not val or not key:
                break
            result[key] = val
            i += 1
        return result

    def insert_after(self, node):
        mupdf.fz_dom_insert_after(self.this, node.this)

    def insert_before(self, node):
        mupdf.fz_dom_insert_before(self.this, node.this)

    def insert_text(self, text):
        lines = text.splitlines()
        line_count = len(lines)
        for i, line in enumerate(lines):
            self.append_child(self.create_text_node(line))
            if i < line_count - 1:
                self.append_child(self.create_element('br'))
        return self

    @property
    def is_text(self):
        """Check if this is a text node."""
        return self.text is not None

    @property
    def last_child(self):
        """Return last child node."""
        child = self.first_child
        if child is None:
            return None
        while True:
            next = child.next
            if not next:
                return child
            child = next

    @property
    def next(self):
        ret = mupdf.fz_dom_next(self.this)
        if ret.m_internal:
            return Xml(ret)

    @property
    def parent(self):
        ret = mupdf.fz_dom_parent(self.this)
        if ret.m_internal:
            return Xml(ret)

    @property
    def previous(self):
        ret = mupdf.fz_dom_previous(self.this)
        if ret.m_internal:
            return Xml(ret)

    def remove(self):
        mupdf.fz_dom_remove(self.this)

    def remove_attribute(self, key):
        assert key
        mupdf.fz_dom_remove_attribute(self.this, key)

    @property
    def root(self):
        return Xml(mupdf.fz_xml_root(self.this))

    def set_align(self, align):
        """Set text alignment via CSS style"""
        text = 'text-align: %s'
        if isinstance(align, str):
            t = align
        elif align == TEXT_ALIGN_LEFT:
            t = 'left'
        elif align == TEXT_ALIGN_CENTER:
            t = 'center'
        elif align == TEXT_ALIGN_RIGHT:
            t = 'right'
        elif align == TEXT_ALIGN_JUSTIFY:
            t = 'justify'
        else:
            raise ValueError(f'Unrecognised align={align!r}')
        text = text % t
        self.add_style(text)
        return self

    def set_attribute(self, key, value):
        assert key
        mupdf.fz_dom_add_attribute(self.this, key, value)

    def set_bgcolor(self, color):
        """Set background color via CSS style"""
        text = f'background-color: %s' % self.color_text(color)
        self.add_style(text)
        return self

    def set_bold(self, val=True):
        """Set bold on / off via CSS style"""
        if val:
            val = 'bold'
        else:
            val = 'normal'
        text = 'font-weight: %s' % val
        self.append_styled_span(text)
        return self

    def set_color(self, color):
        """Set text color via CSS style"""
        text = f'color: %s' % self.color_text(color)
        self.append_styled_span(text)
        return self

    def set_columns(self, cols):
        """Set number of text columns via CSS style"""
        text = f'columns: {cols}'
        self.append_styled_span(text)
        return self

    def set_font(self, font):
        """Set font-family name via CSS style"""
        text = 'font-family: %s' % font
        self.append_styled_span(text)
        return self

    def set_fontsize(self, fontsize):
        """Set font size name via CSS style"""
        if type(fontsize) is str:
            px = ''
        else:
            px = 'px'
        text = f'font-size: {fontsize}{px}'
        self.append_styled_span(text)
        return self

    def set_id(self, unique):
        """Set a unique id."""
        root = self.root
        if root.find(None, 'id', unique):
            raise ValueError(f"id '{unique}' already exists")
        self.set_attribute('id', unique)
        return self

    def set_italic(self, val=True):
        """Set italic on / off via CSS style"""
        if val:
            val = 'italic'
        else:
            val = 'normal'
        text = 'font-style: %s' % val
        self.append_styled_span(text)
        return self

    def set_leading(self, leading):
        """Set inter-line spacing value via CSS style - block-level only."""
        text = f'-mupdf-leading: {leading}'
        self.add_style(text)
        return self

    def set_letter_spacing(self, spacing):
        """Set inter-letter spacing value via CSS style"""
        text = f'letter-spacing: {spacing}'
        self.append_styled_span(text)
        return self

    def set_lineheight(self, lineheight):
        """Set line height name via CSS style - block-level only."""
        text = f'line-height: {lineheight}'
        self.add_style(text)
        return self

    def set_margins(self, val):
        """Set margin values via CSS style"""
        text = 'margins: %s' % val
        self.append_styled_span(text)
        return self

    def set_opacity(self, opacity):
        """Set opacity via CSS style"""
        text = f'opacity: {opacity}'
        self.append_styled_span(text)
        return self

    def set_pagebreak_after(self):
        """Insert a page break after this node."""
        text = 'page-break-after: always'
        self.add_style(text)
        return self

    def set_pagebreak_before(self):
        """Insert a page break before this node."""
        text = 'page-break-before: always'
        self.add_style(text)
        return self

    def set_properties(self, align=None, bgcolor=None, bold=None, color=None, columns=None, font=None, fontsize=None, indent=None, italic=None, leading=None, letter_spacing=None, lineheight=None, margins=None, pagebreak_after=None, pagebreak_before=None, word_spacing=None, unqid=None, cls=None):
        """Set any or all properties of a node.

        To be used for existing nodes preferrably.
        """
        root = self.root
        temp = root.add_division()
        if align is not None:
            temp.set_align(align)
        if bgcolor is not None:
            temp.set_bgcolor(bgcolor)
        if bold is not None:
            temp.set_bold(bold)
        if color is not None:
            temp.set_color(color)
        if columns is not None:
            temp.set_columns(columns)
        if font is not None:
            temp.set_font(font)
        if fontsize is not None:
            temp.set_fontsize(fontsize)
        if indent is not None:
            temp.set_text_indent(indent)
        if italic is not None:
            temp.set_italic(italic)
        if leading is not None:
            temp.set_leading(leading)
        if letter_spacing is not None:
            temp.set_letter_spacing(letter_spacing)
        if lineheight is not None:
            temp.set_lineheight(lineheight)
        if margins is not None:
            temp.set_margins(margins)
        if pagebreak_after is not None:
            temp.set_pagebreak_after()
        if pagebreak_before is not None:
            temp.set_pagebreak_before()
        if word_spacing is not None:
            temp.set_word_spacing(word_spacing)
        if unqid is not None:
            self.set_id(unqid)
        if cls is not None:
            self.add_class(cls)
        styles = []
        top_style = temp.get_attribute_value('style')
        if top_style is not None:
            styles.append(top_style)
        child = temp.first_child
        while child:
            styles.append(child.get_attribute_value('style'))
            child = child.first_child
        self.set_attribute('style', ';'.join(styles))
        temp.remove()
        return self

    def set_text_indent(self, indent):
        """Set text indentation name via CSS style - block-level only."""
        text = f'text-indent: {indent}'
        self.add_style(text)
        return self

    def set_underline(self, val='underline'):
        text = 'text-decoration: %s' % val
        self.append_styled_span(text)
        return self

    def set_word_spacing(self, spacing):
        """Set inter-word spacing value via CSS style"""
        text = f'word-spacing: {spacing}'
        self.append_styled_span(text)
        return self

    def span_bottom(self):
        """Find deepest level in stacked spans."""
        parent = self
        child = self.last_child
        if child is None:
            return None
        while child.is_text:
            child = child.previous
            if child is None:
                break
        if child is None or child.tagname != 'span':
            return None
        while True:
            if child is None:
                return parent
            if child.tagname in ('a', 'sub', 'sup', 'body') or child.is_text:
                child = child.next
                continue
            if child.tagname == 'span':
                parent = child
                child = child.first_child
            else:
                return parent

    @property
    def tagname(self):
        return mupdf.fz_xml_tag(self.this)

    @property
    def text(self):
        return mupdf.fz_xml_text(self.this)
    add_var = add_code
    add_samp = add_code
    add_kbd = add_code