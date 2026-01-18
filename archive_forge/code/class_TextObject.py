from __future__ import unicode_literals
from prompt_toolkit.buffer import ClipboardData, indent, unindent, reshape_text
from prompt_toolkit.document import Document
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER, SYSTEM_BUFFER
from prompt_toolkit.filters import Filter, Condition, HasArg, Always, IsReadOnly
from prompt_toolkit.filters.cli import ViNavigationMode, ViInsertMode, ViInsertMultipleMode, ViReplaceMode, ViSelectionMode, ViWaitingForTextObjectMode, ViDigraphMode, ViMode
from prompt_toolkit.key_binding.digraphs import DIGRAPHS
from prompt_toolkit.key_binding.vi_state import CharacterFind, InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from prompt_toolkit.selection import SelectionType, SelectionState, PasteMode
from .scroll import scroll_forward, scroll_backward, scroll_half_page_up, scroll_half_page_down, scroll_one_line_up, scroll_one_line_down, scroll_page_up, scroll_page_down
from .named_commands import get_by_name
from ..registry import Registry, ConditionalRegistry, BaseRegistry
import prompt_toolkit.filters as filters
from six.moves import range
import codecs
import six
import string
class TextObject(object):
    """
    Return struct for functions wrapped in ``text_object``.
    Both `start` and `end` are relative to the current cursor position.
    """

    def __init__(self, start, end=0, type=TextObjectType.EXCLUSIVE):
        self.start = start
        self.end = end
        self.type = type

    @property
    def selection_type(self):
        if self.type == TextObjectType.LINEWISE:
            return SelectionType.LINES
        if self.type == TextObjectType.BLOCK:
            return SelectionType.BLOCK
        else:
            return SelectionType.CHARACTERS

    def sorted(self):
        """
        Return a (start, end) tuple where start <= end.
        """
        if self.start < self.end:
            return (self.start, self.end)
        else:
            return (self.end, self.start)

    def operator_range(self, document):
        """
        Return a (start, end) tuple with start <= end that indicates the range
        operators should operate on.
        `buffer` is used to get start and end of line positions.
        """
        start, end = self.sorted()
        doc = document
        if self.type == TextObjectType.EXCLUSIVE and doc.translate_index_to_position(end + doc.cursor_position)[1] == 0:
            end -= 1
        if self.type == TextObjectType.INCLUSIVE:
            end += 1
        if self.type == TextObjectType.LINEWISE:
            row, col = doc.translate_index_to_position(start + doc.cursor_position)
            start = doc.translate_row_col_to_index(row, 0) - doc.cursor_position
            row, col = doc.translate_index_to_position(end + doc.cursor_position)
            end = doc.translate_row_col_to_index(row, len(doc.lines[row])) - doc.cursor_position
        return (start, end)

    def get_line_numbers(self, buffer):
        """
        Return a (start_line, end_line) pair.
        """
        from_, to = self.operator_range(buffer.document)
        from_ += buffer.cursor_position
        to += buffer.cursor_position
        from_, _ = buffer.document.translate_index_to_position(from_)
        to, _ = buffer.document.translate_index_to_position(to)
        return (from_, to)

    def cut(self, buffer):
        """
        Turn text object into `ClipboardData` instance.
        """
        from_, to = self.operator_range(buffer.document)
        from_ += buffer.cursor_position
        to += buffer.cursor_position
        to -= 1
        document = Document(buffer.text, to, SelectionState(original_cursor_position=from_, type=self.selection_type))
        new_document, clipboard_data = document.cut_selection()
        return (new_document, clipboard_data)