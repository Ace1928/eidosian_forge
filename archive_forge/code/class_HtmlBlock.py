from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
class HtmlBlock(Block):
    accepts_lines = True

    @staticmethod
    def continue_(parser=None, container=None):
        if parser.blank and (container.html_block_type == 6 or container.html_block_type == 7):
            return 1
        else:
            return 0

    @staticmethod
    def finalize(parser=None, block=None):
        block.literal = re.sub('(\\n *)+$', '', block.string_content)
        block.string_content = None

    @staticmethod
    def can_contain(t):
        return False