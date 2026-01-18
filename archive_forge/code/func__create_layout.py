from __future__ import annotations
from asyncio import get_running_loop
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Generic, Iterator, TypeVar, Union, cast
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggest, DynamicAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import Clipboard, DynamicClipboard, InMemoryClipboard
from prompt_toolkit.completion import Completer, DynamicCompleter, ThreadedCompleter
from prompt_toolkit.cursor_shapes import (
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from prompt_toolkit.eventloop import InputHook
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding.bindings.auto_suggest import load_auto_suggest_bindings
from prompt_toolkit.key_binding.bindings.completion import (
from prompt_toolkit.key_binding.bindings.open_in_editor import (
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer, WindowAlign
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from prompt_toolkit.layout.processors import (
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.lexers import DynamicLexer, Lexer
from prompt_toolkit.output import ColorDepth, DummyOutput, Output
from prompt_toolkit.styles import (
from prompt_toolkit.utils import (
from prompt_toolkit.validation import DynamicValidator, Validator
from prompt_toolkit.widgets.toolbars import (
def _create_layout(self) -> Layout:
    """
        Create `Layout` for this prompt.
        """
    dyncond = self._dyncond
    has_before_fragments, get_prompt_text_1, get_prompt_text_2 = _split_multiline_prompt(self._get_prompt)
    default_buffer = self.default_buffer
    search_buffer = self.search_buffer

    @Condition
    def display_placeholder() -> bool:
        return self.placeholder is not None and self.default_buffer.text == ''
    all_input_processors = [HighlightIncrementalSearchProcessor(), HighlightSelectionProcessor(), ConditionalProcessor(AppendAutoSuggestion(), has_focus(default_buffer) & ~is_done), ConditionalProcessor(PasswordProcessor(), dyncond('is_password')), DisplayMultipleCursors(), DynamicProcessor(lambda: merge_processors(self.input_processors or [])), ConditionalProcessor(AfterInput(lambda: self.placeholder), filter=display_placeholder)]
    bottom_toolbar = ConditionalContainer(Window(FormattedTextControl(lambda: self.bottom_toolbar, style='class:bottom-toolbar.text'), style='class:bottom-toolbar', dont_extend_height=True, height=Dimension(min=1)), filter=Condition(lambda: self.bottom_toolbar is not None) & ~is_done & renderer_height_is_known)
    search_toolbar = SearchToolbar(search_buffer, ignore_case=dyncond('search_ignore_case'))
    search_buffer_control = SearchBufferControl(buffer=search_buffer, input_processors=[ReverseSearchProcessor()], ignore_case=dyncond('search_ignore_case'))
    system_toolbar = SystemToolbar(enable_global_bindings=dyncond('enable_system_prompt'))

    def get_search_buffer_control() -> SearchBufferControl:
        """Return the UIControl to be focused when searching start."""
        if is_true(self.multiline):
            return search_toolbar.control
        else:
            return search_buffer_control
    default_buffer_control = BufferControl(buffer=default_buffer, search_buffer_control=get_search_buffer_control, input_processors=all_input_processors, include_default_input_processors=False, lexer=DynamicLexer(lambda: self.lexer), preview_search=True)
    default_buffer_window = Window(default_buffer_control, height=self._get_default_buffer_control_height, get_line_prefix=partial(self._get_line_prefix, get_prompt_text_2=get_prompt_text_2), wrap_lines=dyncond('wrap_lines'))

    @Condition
    def multi_column_complete_style() -> bool:
        return self.complete_style == CompleteStyle.MULTI_COLUMN
    layout = HSplit([FloatContainer(HSplit([ConditionalContainer(Window(FormattedTextControl(get_prompt_text_1), dont_extend_height=True), Condition(has_before_fragments)), ConditionalContainer(default_buffer_window, Condition(lambda: get_app().layout.current_control != search_buffer_control)), ConditionalContainer(Window(search_buffer_control), Condition(lambda: get_app().layout.current_control == search_buffer_control))]), [Float(xcursor=True, ycursor=True, transparent=True, content=CompletionsMenu(max_height=16, scroll_offset=1, extra_filter=has_focus(default_buffer) & ~multi_column_complete_style)), Float(xcursor=True, ycursor=True, transparent=True, content=MultiColumnCompletionsMenu(show_meta=True, extra_filter=has_focus(default_buffer) & multi_column_complete_style)), Float(right=0, top=0, hide_when_covering_content=True, content=_RPrompt(lambda: self.rprompt))]), ConditionalContainer(ValidationToolbar(), filter=~is_done), ConditionalContainer(system_toolbar, dyncond('enable_system_prompt') & ~is_done), ConditionalContainer(Window(FormattedTextControl(self._get_arg_text), height=1), dyncond('multiline') & has_arg), ConditionalContainer(search_toolbar, dyncond('multiline') & ~is_done), bottom_toolbar])
    return Layout(layout, default_buffer_window)