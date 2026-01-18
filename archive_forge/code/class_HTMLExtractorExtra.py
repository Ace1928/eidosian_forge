from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
class HTMLExtractorExtra(HTMLExtractor):
    """
    Override `HTMLExtractor` and create `etree` `Elements` for any elements which should have content parsed as
    Markdown.
    """

    def __init__(self, md: Markdown, *args, **kwargs):
        self.block_level_tags = set(md.block_level_elements.copy())
        self.span_tags = set(['address', 'dd', 'dt', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'legend', 'li', 'p', 'summary', 'td', 'th'])
        self.raw_tags = set(['canvas', 'math', 'option', 'pre', 'script', 'style', 'textarea'])
        super().__init__(md, *args, **kwargs)
        self.block_tags = set(self.block_level_tags) - (self.span_tags | self.raw_tags | self.empty_tags)
        self.span_and_blocks_tags = self.block_tags | self.span_tags

    def reset(self):
        """Reset this instance.  Loses all unprocessed data."""
        self.mdstack: list[str] = []
        self.treebuilder = etree.TreeBuilder()
        self.mdstate: list[Literal['block', 'span', 'off', None]] = []
        super().reset()

    def close(self):
        """Handle any buffered data."""
        super().close()
        if self.mdstack:
            self.handle_endtag(self.mdstack[0])

    def get_element(self) -> etree.Element:
        """ Return element from `treebuilder` and reset `treebuilder` for later use. """
        element = self.treebuilder.close()
        self.treebuilder = etree.TreeBuilder()
        return element

    def get_state(self, tag, attrs: Mapping[str, str]) -> Literal['block', 'span', 'off', None]:
        """ Return state from tag and `markdown` attribute. One of 'block', 'span', or 'off'. """
        md_attr = attrs.get('markdown', '0')
        if md_attr == 'markdown':
            md_attr = '1'
        parent_state = self.mdstate[-1] if self.mdstate else None
        if parent_state == 'off' or (parent_state == 'span' and md_attr != '0'):
            md_attr = parent_state
        if md_attr == '1' and tag in self.block_tags or (md_attr == 'block' and tag in self.span_and_blocks_tags):
            return 'block'
        elif md_attr == '1' and tag in self.span_tags or (md_attr == 'span' and tag in self.span_and_blocks_tags):
            return 'span'
        elif tag in self.block_level_tags:
            return 'off'
        else:
            return None

    def handle_starttag(self, tag, attrs):
        if tag in self.empty_tags and (self.at_line_start() or self.intail):
            attrs = {key: value if value is not None else key for key, value in attrs}
            if 'markdown' in attrs:
                attrs.pop('markdown')
                element = etree.Element(tag, attrs)
                data = etree.tostring(element, encoding='unicode', method='html')
            else:
                data = self.get_starttag_text()
            self.handle_empty_tag(data, True)
            return
        if tag in self.block_level_tags and (self.at_line_start() or self.intail):
            attrs = {key: value if value is not None else key for key, value in attrs}
            state = self.get_state(tag, attrs)
            if self.inraw or (state in [None, 'off'] and (not self.mdstack)):
                attrs.pop('markdown', None)
                super().handle_starttag(tag, attrs)
            else:
                if 'p' in self.mdstack and tag in self.block_level_tags:
                    self.handle_endtag('p')
                self.mdstate.append(state)
                self.mdstack.append(tag)
                attrs['markdown'] = state
                self.treebuilder.start(tag, attrs)
        elif self.inraw:
            super().handle_starttag(tag, attrs)
        else:
            text = self.get_starttag_text()
            if self.mdstate and self.mdstate[-1] == 'off':
                self.handle_data(self.md.htmlStash.store(text))
            else:
                self.handle_data(text)
            if tag in self.CDATA_CONTENT_ELEMENTS:
                self.clear_cdata_mode()

    def handle_endtag(self, tag):
        if tag in self.block_level_tags:
            if self.inraw:
                super().handle_endtag(tag)
            elif tag in self.mdstack:
                while self.mdstack:
                    item = self.mdstack.pop()
                    self.mdstate.pop()
                    self.treebuilder.end(item)
                    if item == tag:
                        break
                if not self.mdstack:
                    element = self.get_element()
                    item = self.cleandoc[-1] if self.cleandoc else ''
                    if not item.endswith('\n\n') and item.endswith('\n'):
                        self.cleandoc.append('\n')
                    self.cleandoc.append(self.md.htmlStash.store(element))
                    self.cleandoc.append('\n\n')
                    self.state = []
                    if not blank_line_re.match(self.rawdata[self.line_offset + self.offset + len(self.get_endtag_text(tag)):]):
                        self.intail = True
            else:
                text = self.get_endtag_text(tag)
                if self.mdstate and self.mdstate[-1] == 'off':
                    self.handle_data(self.md.htmlStash.store(text))
                else:
                    self.handle_data(text)
        elif self.inraw:
            super().handle_endtag(tag)
        else:
            text = self.get_endtag_text(tag)
            if self.mdstate and self.mdstate[-1] == 'off':
                self.handle_data(self.md.htmlStash.store(text))
            else:
                self.handle_data(text)

    def handle_startendtag(self, tag, attrs):
        if tag in self.empty_tags:
            attrs = {key: value if value is not None else key for key, value in attrs}
            if 'markdown' in attrs:
                attrs.pop('markdown')
                element = etree.Element(tag, attrs)
                data = etree.tostring(element, encoding='unicode', method='html')
            else:
                data = self.get_starttag_text()
        else:
            data = self.get_starttag_text()
        self.handle_empty_tag(data, is_block=self.md.is_block_level(tag))

    def handle_data(self, data):
        if self.intail and '\n' in data:
            self.intail = False
        if self.inraw or not self.mdstack:
            super().handle_data(data)
        else:
            self.treebuilder.data(data)

    def handle_empty_tag(self, data, is_block):
        if self.inraw or not self.mdstack:
            super().handle_empty_tag(data, is_block)
        elif self.at_line_start() and is_block:
            self.handle_data('\n' + self.md.htmlStash.store(data) + '\n\n')
        else:
            self.handle_data(self.md.htmlStash.store(data))

    def parse_pi(self, i: int) -> int:
        if self.at_line_start() or self.intail or self.mdstack:
            return super(HTMLExtractor, self).parse_pi(i)
        self.handle_data('<?')
        return i + 2

    def parse_html_declaration(self, i: int) -> int:
        if self.at_line_start() or self.intail or self.mdstack:
            return super(HTMLExtractor, self).parse_html_declaration(i)
        self.handle_data('<!')
        return i + 2