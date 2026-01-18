from __future__ import annotations
import re
import importlib.util
import sys
from typing import TYPE_CHECKING, Sequence
class HTMLExtractor(htmlparser.HTMLParser):
    """
    Extract raw HTML from text.

    The raw HTML is stored in the [`htmlStash`][markdown.util.HtmlStash] of the
    [`Markdown`][markdown.Markdown] instance passed to `md` and the remaining text
    is stored in `cleandoc` as a list of strings.
    """

    def __init__(self, md: Markdown, *args, **kwargs):
        if 'convert_charrefs' not in kwargs:
            kwargs['convert_charrefs'] = False
        self.empty_tags = set(['hr'])
        self.lineno_start_cache = [0]
        super().__init__(*args, **kwargs)
        self.md = md

    def reset(self):
        """Reset this instance.  Loses all unprocessed data."""
        self.inraw = False
        self.intail = False
        self.stack: list[str] = []
        self._cache: list[str] = []
        self.cleandoc: list[str] = []
        self.lineno_start_cache = [0]
        super().reset()

    def close(self):
        """Handle any buffered data."""
        super().close()
        if len(self.rawdata):
            if self.convert_charrefs and (not self.cdata_elem):
                self.handle_data(htmlparser.unescape(self.rawdata))
            else:
                self.handle_data(self.rawdata)
        if len(self._cache):
            self.cleandoc.append(self.md.htmlStash.store(''.join(self._cache)))
            self._cache = []

    @property
    def line_offset(self) -> int:
        """Returns char index in `self.rawdata` for the start of the current line. """
        for ii in range(len(self.lineno_start_cache) - 1, self.lineno - 1):
            last_line_start_pos = self.lineno_start_cache[ii]
            lf_pos = self.rawdata.find('\n', last_line_start_pos)
            if lf_pos == -1:
                lf_pos = len(self.rawdata)
            self.lineno_start_cache.append(lf_pos + 1)
        return self.lineno_start_cache[self.lineno - 1]

    def at_line_start(self) -> bool:
        """
        Returns True if current position is at start of line.

        Allows for up to three blank spaces at start of line.
        """
        if self.offset == 0:
            return True
        if self.offset > 3:
            return False
        return self.rawdata[self.line_offset:self.line_offset + self.offset].strip() == ''

    def get_endtag_text(self, tag: str) -> str:
        """
        Returns the text of the end tag.

        If it fails to extract the actual text from the raw data, it builds a closing tag with `tag`.
        """
        start = self.line_offset + self.offset
        m = htmlparser.endendtag.search(self.rawdata, start)
        if m:
            return self.rawdata[start:m.end()]
        else:
            return '</{}>'.format(tag)

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, str]]):
        if tag in self.empty_tags:
            self.handle_startendtag(tag, attrs)
            return
        if self.md.is_block_level(tag) and (self.intail or (self.at_line_start() and (not self.inraw))):
            self.inraw = True
            self.cleandoc.append('\n')
        text = self.get_starttag_text()
        if self.inraw:
            self.stack.append(tag)
            self._cache.append(text)
        else:
            self.cleandoc.append(text)
            if tag in self.CDATA_CONTENT_ELEMENTS:
                self.clear_cdata_mode()

    def handle_endtag(self, tag: str):
        text = self.get_endtag_text(tag)
        if self.inraw:
            self._cache.append(text)
            if tag in self.stack:
                while self.stack:
                    if self.stack.pop() == tag:
                        break
            if len(self.stack) == 0:
                if blank_line_re.match(self.rawdata[self.line_offset + self.offset + len(text):]):
                    self._cache.append('\n')
                else:
                    self.intail = True
                self.inraw = False
                self.cleandoc.append(self.md.htmlStash.store(''.join(self._cache)))
                self.cleandoc.append('\n\n')
                self._cache = []
        else:
            self.cleandoc.append(text)

    def handle_data(self, data: str):
        if self.intail and '\n' in data:
            self.intail = False
        if self.inraw:
            self._cache.append(data)
        else:
            self.cleandoc.append(data)

    def handle_empty_tag(self, data: str, is_block: bool):
        """ Handle empty tags (`<data>`). """
        if self.inraw or self.intail:
            self._cache.append(data)
        elif self.at_line_start() and is_block:
            if blank_line_re.match(self.rawdata[self.line_offset + self.offset + len(data):]):
                data += '\n'
            else:
                self.intail = True
            item = self.cleandoc[-1] if self.cleandoc else ''
            if not item.endswith('\n\n') and item.endswith('\n'):
                self.cleandoc.append('\n')
            self.cleandoc.append(self.md.htmlStash.store(data))
            self.cleandoc.append('\n\n')
        else:
            self.cleandoc.append(data)

    def handle_startendtag(self, tag: str, attrs):
        self.handle_empty_tag(self.get_starttag_text(), is_block=self.md.is_block_level(tag))

    def handle_charref(self, name: str):
        self.handle_empty_tag('&#{};'.format(name), is_block=False)

    def handle_entityref(self, name: str):
        self.handle_empty_tag('&{};'.format(name), is_block=False)

    def handle_comment(self, data: str):
        self.handle_empty_tag('<!--{}-->'.format(data), is_block=True)

    def handle_decl(self, data: str):
        self.handle_empty_tag('<!{}>'.format(data), is_block=True)

    def handle_pi(self, data: str):
        self.handle_empty_tag('<?{}?>'.format(data), is_block=True)

    def unknown_decl(self, data: str):
        end = ']]>' if data.startswith('CDATA[') else ']>'
        self.handle_empty_tag('<![{}{}'.format(data, end), is_block=True)

    def parse_pi(self, i: int) -> int:
        if self.at_line_start() or self.intail:
            return super().parse_pi(i)
        self.handle_data('<?')
        return i + 2

    def parse_html_declaration(self, i: int) -> int:
        if self.at_line_start() or self.intail:
            return super().parse_html_declaration(i)
        self.handle_data('<!')
        return i + 2

    def parse_bogus_comment(self, i: int, report: int=0) -> int:
        pos = super().parse_bogus_comment(i, report)
        if pos == -1:
            return -1
        self.handle_empty_tag(self.rawdata[i:pos], is_block=False)
        return pos
    __starttag_text: str | None = None

    def get_starttag_text(self) -> str:
        """Return full source of start tag: `<...>`."""
        return self.__starttag_text

    def parse_starttag(self, i: int) -> int:
        self.__starttag_text = None
        endpos = self.check_for_whole_start_tag(i)
        if endpos < 0:
            return endpos
        rawdata = self.rawdata
        self.__starttag_text = rawdata[i:endpos]
        attrs = []
        match = htmlparser.tagfind_tolerant.match(rawdata, i + 1)
        assert match, 'unexpected call to parse_starttag()'
        k = match.end()
        self.lasttag = tag = match.group(1).lower()
        while k < endpos:
            m = htmlparser.attrfind_tolerant.match(rawdata, k)
            if not m:
                break
            attrname, rest, attrvalue = m.group(1, 2, 3)
            if not rest:
                attrvalue = None
            elif attrvalue[:1] == "'" == attrvalue[-1:] or attrvalue[:1] == '"' == attrvalue[-1:]:
                attrvalue = attrvalue[1:-1]
            if attrvalue:
                attrvalue = htmlparser.unescape(attrvalue)
            attrs.append((attrname.lower(), attrvalue))
            k = m.end()
        end = rawdata[k:endpos].strip()
        if end not in ('>', '/>'):
            lineno, offset = self.getpos()
            if '\n' in self.__starttag_text:
                lineno = lineno + self.__starttag_text.count('\n')
                offset = len(self.__starttag_text) - self.__starttag_text.rfind('\n')
            else:
                offset = offset + len(self.__starttag_text)
            self.handle_data(rawdata[i:endpos])
            return endpos
        if end.endswith('/>'):
            self.handle_startendtag(tag, attrs)
        else:
            if tag in self.CDATA_CONTENT_ELEMENTS:
                self.set_cdata_mode(tag)
            self.handle_starttag(tag, attrs)
        return endpos