from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from six import with_metaclass
from six.moves import range
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.search_state import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .lexers import Lexer, SimpleLexer
from .processors import Processor
from .screen import Char, Point
from .utils import token_list_width, split_lines, token_list_to_text
import six
import time
class BufferControl(UIControl):
    """
    Control for visualising the content of a `Buffer`.

    :param input_processors: list of :class:`~prompt_toolkit.layout.processors.Processor`.
    :param lexer: :class:`~prompt_toolkit.layout.lexers.Lexer` instance for syntax highlighting.
    :param preview_search: `bool` or `CLIFilter`: Show search while typing.
    :param get_search_state: Callable that takes a CommandLineInterface and
        returns the SearchState to be used. (If not CommandLineInterface.search_state.)
    :param buffer_name: String representing the name of the buffer to display.
    :param default_char: :class:`.Char` instance to use to fill the background. This is
        transparent by default.
    :param focus_on_click: Focus this buffer when it's click, but not yet focussed.
    """

    def __init__(self, buffer_name=DEFAULT_BUFFER, input_processors=None, lexer=None, preview_search=False, search_buffer_name=SEARCH_BUFFER, get_search_state=None, menu_position=None, default_char=None, focus_on_click=False):
        assert input_processors is None or all((isinstance(i, Processor) for i in input_processors))
        assert menu_position is None or callable(menu_position)
        assert lexer is None or isinstance(lexer, Lexer)
        assert get_search_state is None or callable(get_search_state)
        assert default_char is None or isinstance(default_char, Char)
        self.preview_search = to_cli_filter(preview_search)
        self.get_search_state = get_search_state
        self.focus_on_click = to_cli_filter(focus_on_click)
        self.input_processors = input_processors or []
        self.buffer_name = buffer_name
        self.menu_position = menu_position
        self.lexer = lexer or SimpleLexer()
        self.default_char = default_char or Char(token=Token.Transparent)
        self.search_buffer_name = search_buffer_name
        self._token_cache = SimpleCache(maxsize=8)
        self._xy_to_cursor_position = None
        self._last_click_timestamp = None
        self._last_get_processed_line = None

    def _buffer(self, cli):
        """
        The buffer object that contains the 'main' content.
        """
        return cli.buffers[self.buffer_name]

    def has_focus(self, cli):
        return cli.current_buffer_name == self.buffer_name or any((i.has_focus(cli) for i in self.input_processors))

    def preferred_width(self, cli, max_available_width):
        """
        This should return the preferred width.

        Note: We don't specify a preferred width according to the content,
              because it would be too expensive. Calculating the preferred
              width can be done by calculating the longest line, but this would
              require applying all the processors to each line. This is
              unfeasible for a larger document, and doing it for small
              documents only would result in inconsistent behaviour.
        """
        return None

    def preferred_height(self, cli, width, max_available_height, wrap_lines):
        height = 0
        content = self.create_content(cli, width, None)
        if not wrap_lines:
            return content.line_count
        if content.line_count >= max_available_height:
            return max_available_height
        for i in range(content.line_count):
            height += content.get_height_for_line(i, width)
            if height >= max_available_height:
                return max_available_height
        return height

    def _get_tokens_for_line_func(self, cli, document):
        """
        Create a function that returns the tokens for a given line.
        """

        def get_tokens_for_line():
            return self.lexer.lex_document(cli, document)
        return self._token_cache.get(document.text, get_tokens_for_line)

    def _create_get_processed_line_func(self, cli, document):
        """
        Create a function that takes a line number of the current document and
        returns a _ProcessedLine(processed_tokens, source_to_display, display_to_source)
        tuple.
        """

        def transform(lineno, tokens):
            """ Transform the tokens for a given line number. """
            source_to_display_functions = []
            display_to_source_functions = []
            if document.cursor_position_row == lineno:
                cursor_column = document.cursor_position_col
            else:
                cursor_column = None

            def source_to_display(i):
                """ Translate x position from the buffer to the x position in the
                processed token list. """
                for f in source_to_display_functions:
                    i = f(i)
                return i
            for p in self.input_processors:
                transformation = p.apply_transformation(cli, document, lineno, source_to_display, tokens)
                tokens = transformation.tokens
                if cursor_column:
                    cursor_column = transformation.source_to_display(cursor_column)
                display_to_source_functions.append(transformation.display_to_source)
                source_to_display_functions.append(transformation.source_to_display)

            def display_to_source(i):
                for f in reversed(display_to_source_functions):
                    i = f(i)
                return i
            return _ProcessedLine(tokens, source_to_display, display_to_source)

        def create_func():
            get_line = self._get_tokens_for_line_func(cli, document)
            cache = {}

            def get_processed_line(i):
                try:
                    return cache[i]
                except KeyError:
                    processed_line = transform(i, get_line(i))
                    cache[i] = processed_line
                    return processed_line
            return get_processed_line
        return create_func()

    def create_content(self, cli, width, height):
        """
        Create a UIContent.
        """
        buffer = self._buffer(cli)

        def preview_now():
            """ True when we should preview a search. """
            return bool(self.preview_search(cli) and cli.buffers[self.search_buffer_name].text)
        if preview_now():
            if self.get_search_state:
                ss = self.get_search_state(cli)
            else:
                ss = cli.search_state
            document = buffer.document_for_search(SearchState(text=cli.current_buffer.text, direction=ss.direction, ignore_case=ss.ignore_case))
        else:
            document = buffer.document
        get_processed_line = self._create_get_processed_line_func(cli, document)
        self._last_get_processed_line = get_processed_line

        def translate_rowcol(row, col):
            """ Return the content column for this coordinate. """
            return Point(y=row, x=get_processed_line(row).source_to_display(col))

        def get_line(i):
            """ Return the tokens for a given line number. """
            tokens = get_processed_line(i).tokens
            tokens = tokens + [(self.default_char.token, ' ')]
            return tokens
        content = UIContent(get_line=get_line, line_count=document.line_count, cursor_position=translate_rowcol(document.cursor_position_row, document.cursor_position_col), default_char=self.default_char)
        if cli.current_buffer_name == self.buffer_name:
            menu_position = self.menu_position(cli) if self.menu_position else None
            if menu_position is not None:
                assert isinstance(menu_position, int)
                menu_row, menu_col = buffer.document.translate_index_to_position(menu_position)
                content.menu_position = translate_rowcol(menu_row, menu_col)
            elif buffer.complete_state:
                menu_row, menu_col = buffer.document.translate_index_to_position(min(buffer.cursor_position, buffer.complete_state.original_document.cursor_position))
                content.menu_position = translate_rowcol(menu_row, menu_col)
            else:
                content.menu_position = None
        return content

    def mouse_handler(self, cli, mouse_event):
        """
        Mouse handler for this control.
        """
        buffer = self._buffer(cli)
        position = mouse_event.position
        if self.has_focus(cli):
            if self._last_get_processed_line:
                processed_line = self._last_get_processed_line(position.y)
                xpos = processed_line.display_to_source(position.x)
                index = buffer.document.translate_row_col_to_index(position.y, xpos)
                if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
                    buffer.exit_selection()
                    buffer.cursor_position = index
                elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                    if abs(buffer.cursor_position - index) > 1:
                        buffer.start_selection(selection_type=SelectionType.CHARACTERS)
                        buffer.cursor_position = index
                    double_click = self._last_click_timestamp and time.time() - self._last_click_timestamp < 0.3
                    self._last_click_timestamp = time.time()
                    if double_click:
                        start, end = buffer.document.find_boundaries_of_current_word()
                        buffer.cursor_position += start
                        buffer.start_selection(selection_type=SelectionType.CHARACTERS)
                        buffer.cursor_position += end - start
                else:
                    return NotImplemented
        elif self.focus_on_click(cli) and mouse_event.event_type == MouseEventType.MOUSE_UP:
            cli.focus(self.buffer_name)
        else:
            return NotImplemented

    def move_cursor_down(self, cli):
        b = self._buffer(cli)
        b.cursor_position += b.document.get_cursor_down_position()

    def move_cursor_up(self, cli):
        b = self._buffer(cli)
        b.cursor_position += b.document.get_cursor_up_position()