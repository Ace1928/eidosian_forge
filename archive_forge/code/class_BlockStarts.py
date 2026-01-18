from __future__ import absolute_import, unicode_literals
import re
from commonmark import common
from commonmark.common import unescape_string
from commonmark.inlines import InlineParser
from commonmark.node import Node
class BlockStarts(object):
    """Block start functions.

    Return values:
    0 = no match
    1 = matched container, keep going
    2 = matched leaf, no more block starts
    """
    METHODS = ['block_quote', 'atx_heading', 'fenced_code_block', 'html_block', 'setext_heading', 'thematic_break', 'list_item', 'indented_code_block']

    @staticmethod
    def block_quote(parser, container=None):
        if not parser.indented and peek(parser.current_line, parser.next_nonspace) == '>':
            parser.advance_next_nonspace()
            parser.advance_offset(1, False)
            if is_space_or_tab(peek(parser.current_line, parser.offset)):
                parser.advance_offset(1, True)
            parser.close_unmatched_blocks()
            parser.add_child('block_quote', parser.next_nonspace)
            return 1
        return 0

    @staticmethod
    def atx_heading(parser, container=None):
        if not parser.indented:
            m = re.search(reATXHeadingMarker, parser.current_line[parser.next_nonspace:])
            if m:
                parser.advance_next_nonspace()
                parser.advance_offset(len(m.group()), False)
                parser.close_unmatched_blocks()
                container = parser.add_child('heading', parser.next_nonspace)
                container.level = len(m.group().strip())
                container.string_content = re.sub('[ \\t]+#+[ \\t]*$', '', re.sub('^[ \\t]*#+[ \\t]*$', '', parser.current_line[parser.offset:]))
                parser.advance_offset(len(parser.current_line) - parser.offset, False)
                return 2
        return 0

    @staticmethod
    def fenced_code_block(parser, container=None):
        if not parser.indented:
            m = re.search(reCodeFence, parser.current_line[parser.next_nonspace:])
            if m:
                fence_length = len(m.group())
                parser.close_unmatched_blocks()
                container = parser.add_child('code_block', parser.next_nonspace)
                container.is_fenced = True
                container.fence_length = fence_length
                container.fence_char = m.group()[0]
                container.fence_offset = parser.indent
                parser.advance_next_nonspace()
                parser.advance_offset(fence_length, False)
                return 2
        return 0

    @staticmethod
    def html_block(parser, container=None):
        if not parser.indented and peek(parser.current_line, parser.next_nonspace) == '<':
            s = parser.current_line[parser.next_nonspace:]
            for block_type in range(1, 8):
                if re.search(reHtmlBlockOpen[block_type], s) and (block_type < 7 or container.t != 'paragraph'):
                    parser.close_unmatched_blocks()
                    b = parser.add_child('html_block', parser.offset)
                    b.html_block_type = block_type
                    return 2
        return 0

    @staticmethod
    def setext_heading(parser, container=None):
        if not parser.indented and container.t == 'paragraph':
            m = re.search(reSetextHeadingLine, parser.current_line[parser.next_nonspace:])
            if m:
                parser.close_unmatched_blocks()
                while peek(container.string_content, 0) == '[':
                    pos = parser.inline_parser.parseReference(container.string_content, parser.refmap)
                    if not pos:
                        break
                    container.string_content = container.string_content[pos:]
                if container.string_content:
                    heading = Node('heading', container.sourcepos)
                    heading.level = 1 if m.group()[0] == '=' else 2
                    heading.string_content = container.string_content
                    container.insert_after(heading)
                    container.unlink()
                    parser.tip = heading
                    parser.advance_offset(len(parser.current_line) - parser.offset, False)
                    return 2
                else:
                    return 0
        return 0

    @staticmethod
    def thematic_break(parser, container=None):
        if not parser.indented and re.search(reThematicBreak, parser.current_line[parser.next_nonspace:]):
            parser.close_unmatched_blocks()
            parser.add_child('thematic_break', parser.next_nonspace)
            parser.advance_offset(len(parser.current_line) - parser.offset, False)
            return 2
        return 0

    @staticmethod
    def list_item(parser, container=None):
        if not parser.indented or container.t == 'list':
            data = parse_list_marker(parser, container)
            if data:
                parser.close_unmatched_blocks()
                if parser.tip.t != 'list' or not lists_match(container.list_data, data):
                    container = parser.add_child('list', parser.next_nonspace)
                    container.list_data = data
                container = parser.add_child('item', parser.next_nonspace)
                container.list_data = data
                return 1
        return 0

    @staticmethod
    def indented_code_block(parser, container=None):
        if parser.indented and parser.tip.t != 'paragraph' and (not parser.blank):
            parser.advance_offset(CODE_INDENT, True)
            parser.close_unmatched_blocks()
            parser.add_child('code_block', parser.offset)
            return 2
        return 0