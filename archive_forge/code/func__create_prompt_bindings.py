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
def _create_prompt_bindings(self) -> KeyBindings:
    """
        Create the KeyBindings for a prompt application.
        """
    kb = KeyBindings()
    handle = kb.add
    default_focused = has_focus(DEFAULT_BUFFER)

    @Condition
    def do_accept() -> bool:
        return not is_true(self.multiline) and self.app.layout.has_focus(DEFAULT_BUFFER)

    @handle('enter', filter=do_accept & default_focused)
    def _accept_input(event: E) -> None:
        """Accept input when enter has been pressed."""
        self.default_buffer.validate_and_handle()

    @Condition
    def readline_complete_style() -> bool:
        return self.complete_style == CompleteStyle.READLINE_LIKE

    @handle('tab', filter=readline_complete_style & default_focused)
    def _complete_like_readline(event: E) -> None:
        """Display completions (like Readline)."""
        display_completions_like_readline(event)

    @handle('c-c', filter=default_focused)
    @handle('<sigint>')
    def _keyboard_interrupt(event: E) -> None:
        """Abort when Control-C has been pressed."""
        event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

    @Condition
    def ctrl_d_condition() -> bool:
        """Ctrl-D binding is only active when the default buffer is selected
            and empty."""
        app = get_app()
        return app.current_buffer.name == DEFAULT_BUFFER and (not app.current_buffer.text)

    @handle('c-d', filter=ctrl_d_condition & default_focused)
    def _eof(event: E) -> None:
        """Exit when Control-D has been pressed."""
        event.app.exit(exception=EOFError, style='class:exiting')
    suspend_supported = Condition(suspend_to_background_supported)

    @Condition
    def enable_suspend() -> bool:
        return to_filter(self.enable_suspend)()

    @handle('c-z', filter=suspend_supported & enable_suspend)
    def _suspend(event: E) -> None:
        """
            Suspend process to background.
            """
        event.app.suspend_to_background()
    return kb