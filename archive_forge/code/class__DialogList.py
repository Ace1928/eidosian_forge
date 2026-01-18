from __future__ import annotations
from functools import partial
from typing import Callable, Generic, Sequence, TypeVar
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggest, DynamicAutoSuggest
from prompt_toolkit.buffer import Buffer, BufferAcceptHandler
from prompt_toolkit.completion import Completer, DynamicCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.history import History
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import AnyDimension, to_dimension
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.margins import (
from prompt_toolkit.layout.processors import (
from prompt_toolkit.lexers import DynamicLexer, Lexer
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.validation import DynamicValidator, Validator
from .toolbars import SearchToolbar
class _DialogList(Generic[_T]):
    """
    Common code for `RadioList` and `CheckboxList`.
    """
    open_character: str = ''
    close_character: str = ''
    container_style: str = ''
    default_style: str = ''
    selected_style: str = ''
    checked_style: str = ''
    multiple_selection: bool = False
    show_scrollbar: bool = True

    def __init__(self, values: Sequence[tuple[_T, AnyFormattedText]], default_values: Sequence[_T] | None=None) -> None:
        assert len(values) > 0
        default_values = default_values or []
        self.values = values
        keys: list[_T] = [value for value, _ in values]
        self.current_values: list[_T] = [value for value in default_values if value in keys]
        self.current_value: _T = default_values[0] if len(default_values) and default_values[0] in keys else values[0][0]
        if len(self.current_values) > 0:
            self._selected_index = keys.index(self.current_values[0])
        else:
            self._selected_index = 0
        kb = KeyBindings()

        @kb.add('up')
        def _up(event: E) -> None:
            self._selected_index = max(0, self._selected_index - 1)

        @kb.add('down')
        def _down(event: E) -> None:
            self._selected_index = min(len(self.values) - 1, self._selected_index + 1)

        @kb.add('pageup')
        def _pageup(event: E) -> None:
            w = event.app.layout.current_window
            if w.render_info:
                self._selected_index = max(0, self._selected_index - len(w.render_info.displayed_lines))

        @kb.add('pagedown')
        def _pagedown(event: E) -> None:
            w = event.app.layout.current_window
            if w.render_info:
                self._selected_index = min(len(self.values) - 1, self._selected_index + len(w.render_info.displayed_lines))

        @kb.add('enter')
        @kb.add(' ')
        def _click(event: E) -> None:
            self._handle_enter()

        @kb.add(Keys.Any)
        def _find(event: E) -> None:
            values = list(self.values)
            for value in values[self._selected_index + 1:] + values:
                text = fragment_list_to_text(to_formatted_text(value[1])).lower()
                if text.startswith(event.data.lower()):
                    self._selected_index = self.values.index(value)
                    return
        self.control = FormattedTextControl(self._get_text_fragments, key_bindings=kb, focusable=True)
        self.window = Window(content=self.control, style=self.container_style, right_margins=[ConditionalMargin(margin=ScrollbarMargin(display_arrows=True), filter=Condition(lambda: self.show_scrollbar))], dont_extend_height=True)

    def _handle_enter(self) -> None:
        if self.multiple_selection:
            val = self.values[self._selected_index][0]
            if val in self.current_values:
                self.current_values.remove(val)
            else:
                self.current_values.append(val)
        else:
            self.current_value = self.values[self._selected_index][0]

    def _get_text_fragments(self) -> StyleAndTextTuples:

        def mouse_handler(mouse_event: MouseEvent) -> None:
            """
            Set `_selected_index` and `current_value` according to the y
            position of the mouse click event.
            """
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                self._selected_index = mouse_event.position.y
                self._handle_enter()
        result: StyleAndTextTuples = []
        for i, value in enumerate(self.values):
            if self.multiple_selection:
                checked = value[0] in self.current_values
            else:
                checked = value[0] == self.current_value
            selected = i == self._selected_index
            style = ''
            if checked:
                style += ' ' + self.checked_style
            if selected:
                style += ' ' + self.selected_style
            result.append((style, self.open_character))
            if selected:
                result.append(('[SetCursorPosition]', ''))
            if checked:
                result.append((style, '*'))
            else:
                result.append((style, ' '))
            result.append((style, self.close_character))
            result.append((self.default_style, ' '))
            result.extend(to_formatted_text(value[1], style=self.default_style))
            result.append(('', '\n'))
        for i in range(len(result)):
            result[i] = (result[i][0], result[i][1], mouse_handler)
        result.pop()
        return result

    def __pt_container__(self) -> Container:
        return self.window