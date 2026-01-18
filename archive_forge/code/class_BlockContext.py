import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library
class BlockContext:

    def __init__(self):
        self.blocks = defaultdict(list)

    def __repr__(self):
        return f'<{self.__class__.__qualname__}: blocks={self.blocks!r}>'

    def add_blocks(self, blocks):
        for name, block in blocks.items():
            self.blocks[name].insert(0, block)

    def pop(self, name):
        try:
            return self.blocks[name].pop()
        except IndexError:
            return None

    def push(self, name, block):
        self.blocks[name].append(block)

    def get_block(self, name):
        try:
            return self.blocks[name][-1]
        except IndexError:
            return None