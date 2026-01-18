from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
class StructuredTextDecoder(pyglet.text.DocumentDecoder):

    def decode(self, text, location=None):
        self.len_text = 0
        self.current_style = {}
        self.next_style = {}
        self.stack = []
        self.list_stack = []
        self.document = pyglet.text.document.FormattedDocument()
        if location is None:
            location = pyglet.resource.FileLocation('')
        self.decode_structured(text, location)
        return self.document

    def decode_structured(self, text, location):
        raise NotImplementedError('abstract')

    def push_style(self, key, styles):
        old_styles = {}
        for name in styles.keys():
            old_styles[name] = self.current_style.get(name)
        self.stack.append((key, old_styles))
        self.current_style.update(styles)
        self.next_style.update(styles)

    def pop_style(self, key):
        for match, _ in self.stack:
            if key == match:
                break
        else:
            return
        while True:
            match, old_styles = self.stack.pop()
            self.next_style.update(old_styles)
            self.current_style.update(old_styles)
            if match == key:
                break

    def add_text(self, text):
        self.document.insert_text(self.len_text, text, self.next_style)
        self.next_style.clear()
        self.len_text += len(text)

    def add_element(self, element):
        self.document.insert_element(self.len_text, element, self.next_style)
        self.next_style.clear()
        self.len_text += 1