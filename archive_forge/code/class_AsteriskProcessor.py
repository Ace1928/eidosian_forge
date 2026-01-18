from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class AsteriskProcessor(InlineProcessor):
    """Emphasis processor for handling strong and em matches inside asterisks."""
    PATTERNS = [EmStrongItem(re.compile(EM_STRONG_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'), EmStrongItem(re.compile(STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'), EmStrongItem(re.compile(STRONG_EM3_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'), EmStrongItem(re.compile(STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'), EmStrongItem(re.compile(EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')]
    ' The various strong and emphasis patterns handled by this processor. '

    def build_single(self, m: re.Match[str], tag: str, idx: int) -> etree.Element:
        """Return single tag."""
        el1 = etree.Element(tag)
        text = m.group(2)
        self.parse_sub_patterns(text, el1, None, idx)
        return el1

    def build_double(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
        """Return double tag."""
        tag1, tag2 = tags.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.Element(tag2)
        text = m.group(2)
        self.parse_sub_patterns(text, el2, None, idx)
        el1.append(el2)
        if len(m.groups()) == 3:
            text = m.group(3)
            self.parse_sub_patterns(text, el1, el2, idx)
        return el1

    def build_double2(self, m: re.Match[str], tags: str, idx: int) -> etree.Element:
        """Return double tags (variant 2): `<strong>text <em>text</em></strong>`."""
        tag1, tag2 = tags.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.Element(tag2)
        text = m.group(2)
        self.parse_sub_patterns(text, el1, None, idx)
        text = m.group(3)
        el1.append(el2)
        self.parse_sub_patterns(text, el2, None, idx)
        return el1

    def parse_sub_patterns(self, data: str, parent: etree.Element, last: etree.Element | None, idx: int) -> None:
        """
        Parses sub patterns.

        `data`: text to evaluate.

        `parent`: Parent to attach text and sub elements to.

        `last`: Last appended child to parent. Can also be None if parent has no children.

        `idx`: Current pattern index that was used to evaluate the parent.
        """
        offset = 0
        pos = 0
        length = len(data)
        while pos < length:
            if self.compiled_re.match(data, pos):
                matched = False
                for index, item in enumerate(self.PATTERNS):
                    if index <= idx:
                        continue
                    m = item.pattern.match(data, pos)
                    if m:
                        text = data[offset:m.start(0)]
                        if text:
                            if last is not None:
                                last.tail = text
                            else:
                                parent.text = text
                        el = self.build_element(m, item.builder, item.tags, index)
                        parent.append(el)
                        last = el
                        offset = pos = m.end(0)
                        matched = True
                if not matched:
                    pos += 1
            else:
                pos += 1
        text = data[offset:]
        if text:
            if last is not None:
                last.tail = text
            else:
                parent.text = text

    def build_element(self, m: re.Match[str], builder: str, tags: str, index: int) -> etree.Element:
        """Element builder."""
        if builder == 'double2':
            return self.build_double2(m, tags, index)
        elif builder == 'double':
            return self.build_double(m, tags, index)
        else:
            return self.build_single(m, tags, index)

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """Parse patterns."""
        el = None
        start = None
        end = None
        for index, item in enumerate(self.PATTERNS):
            m1 = item.pattern.match(data, m.start(0))
            if m1:
                start = m1.start(0)
                end = m1.end(0)
                el = self.build_element(m1, item.builder, item.tags, index)
                break
        return (el, start, end)