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
@contextmanager
def _dumb_prompt(self, message: AnyFormattedText='') -> Iterator[Application[_T]]:
    """
        Create prompt `Application` for prompt function for dumb terminals.

        Dumb terminals have minimum rendering capabilities. We can only print
        text to the screen. We can't use colors, and we can't do cursor
        movements. The Emacs inferior shell is an example of a dumb terminal.

        We will show the prompt, and wait for the input. We still handle arrow
        keys, and all custom key bindings, but we don't really render the
        cursor movements. Instead we only print the typed character that's
        right before the cursor.
        """
    self.output.write(fragment_list_to_text(to_formatted_text(self.message)))
    self.output.flush()
    key_bindings: KeyBindingsBase = self._create_prompt_bindings()
    if self.key_bindings:
        key_bindings = merge_key_bindings([self.key_bindings, key_bindings])
    application = cast(Application[_T], Application(input=self.input, output=DummyOutput(), layout=self.layout, key_bindings=key_bindings))

    def on_text_changed(_: object) -> None:
        self.output.write(self.default_buffer.document.text_before_cursor[-1:])
        self.output.flush()
    self.default_buffer.on_text_changed += on_text_changed
    try:
        yield application
    finally:
        self.output.write('\r\n')
        self.output.flush()
        self.default_buffer.on_text_changed -= on_text_changed