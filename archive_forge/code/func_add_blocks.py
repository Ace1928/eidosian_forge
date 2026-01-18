import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library
def add_blocks(self, blocks):
    for name, block in blocks.items():
        self.blocks[name].insert(0, block)