import re
from typing import Dict, Any
class BlockState:
    """The state to save block parser's cursor and tokens."""

    def __init__(self, parent=None):
        self.src = ''
        self.tokens = []
        self.cursor = 0
        self.cursor_max = 0
        self.list_tight = True
        self.parent = parent
        if parent:
            self.env = parent.env
        else:
            self.env = {'ref_links': {}}

    def child_state(self, src: str):
        child = self.__class__(self)
        child.process(src)
        return child

    def process(self, src: str):
        self.src = src
        self.cursor_max = len(src)

    def find_line_end(self):
        m = _LINE_END.search(self.src, self.cursor)
        return m.end()

    def get_text(self, end_pos: int):
        return self.src[self.cursor:end_pos]

    def last_token(self):
        if self.tokens:
            return self.tokens[-1]

    def prepend_token(self, token: Dict[str, Any]):
        """Insert token before the last token."""
        self.tokens.insert(len(self.tokens) - 1, token)

    def append_token(self, token: Dict[str, Any]):
        """Add token to the end of token list."""
        self.tokens.append(token)

    def add_paragraph(self, text: str):
        last_token = self.last_token()
        if last_token and last_token['type'] == 'paragraph':
            last_token['text'] += text
        else:
            self.tokens.append({'type': 'paragraph', 'text': text})

    def append_paragraph(self):
        last_token = self.last_token()
        if last_token and last_token['type'] == 'paragraph':
            pos = self.find_line_end()
            last_token['text'] += self.get_text(pos)
            return pos

    def depth(self):
        d = 0
        parent = self.parent
        while parent:
            d += 1
            parent = parent.parent
        return d