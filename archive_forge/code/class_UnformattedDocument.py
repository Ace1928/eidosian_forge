from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
class UnformattedDocument(AbstractDocument):
    """A document having uniform style over all text.

    Changes to the style of text within the document affects the entire
    document.  For convenience, the ``position`` parameters of the style
    methods may therefore be omitted.
    """

    def __init__(self, text=''):
        super().__init__(text)
        self.styles = {}

    def get_style_runs(self, attribute):
        value = self.styles.get(attribute)
        return runlist.ConstRunIterator(len(self.text), value)

    def get_style(self, attribute, position=None):
        return self.styles.get(attribute)

    def set_style(self, start, end, attributes):
        return super().set_style(0, len(self.text), attributes)

    def _set_style(self, start, end, attributes):
        self.styles.update(attributes)

    def set_paragraph_style(self, start, end, attributes):
        return super().set_paragraph_style(0, len(self.text), attributes)

    def get_font_runs(self, dpi=None):
        ft = self.get_font(dpi=dpi)
        return runlist.ConstRunIterator(len(self.text), ft)

    def get_font(self, position=None, dpi=None):
        from pyglet import font
        font_name = self.styles.get('font_name')
        font_size = self.styles.get('font_size')
        bold = self.styles.get('bold', False)
        italic = self.styles.get('italic', False)
        stretch = self.styles.get('stretch', False)
        return font.load(font_name, font_size, bold=bold, italic=italic, stretch=stretch, dpi=dpi)

    def get_element_runs(self):
        return runlist.ConstRunIterator(len(self._text), None)