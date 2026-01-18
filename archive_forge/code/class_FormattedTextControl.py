from __future__ import annotations
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, Iterable, NamedTuple
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.lexers import Lexer, SimpleLexer
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.search import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.utils import get_cwidth
from .processors import (
class FormattedTextControl(UIControl):
    """
    Control that displays formatted text. This can be either plain text, an
    :class:`~prompt_toolkit.formatted_text.HTML` object an
    :class:`~prompt_toolkit.formatted_text.ANSI` object, a list of ``(style_str,
    text)`` tuples or a callable that takes no argument and returns one of
    those, depending on how you prefer to do the formatting. See
    ``prompt_toolkit.layout.formatted_text`` for more information.

    (It's mostly optimized for rather small widgets, like toolbars, menus, etc...)

    When this UI control has the focus, the cursor will be shown in the upper
    left corner of this control by default. There are two ways for specifying
    the cursor position:

    - Pass a `get_cursor_position` function which returns a `Point` instance
      with the current cursor position.

    - If the (formatted) text is passed as a list of ``(style, text)`` tuples
      and there is one that looks like ``('[SetCursorPosition]', '')``, then
      this will specify the cursor position.

    Mouse support:

        The list of fragments can also contain tuples of three items, looking like:
        (style_str, text, handler). When mouse support is enabled and the user
        clicks on this fragment, then the given handler is called. That handler
        should accept two inputs: (Application, MouseEvent) and it should
        either handle the event or return `NotImplemented` in case we want the
        containing Window to handle this event.

    :param focusable: `bool` or :class:`.Filter`: Tell whether this control is
        focusable.

    :param text: Text or formatted text to be displayed.
    :param style: Style string applied to the content. (If you want to style
        the whole :class:`~prompt_toolkit.layout.Window`, pass the style to the
        :class:`~prompt_toolkit.layout.Window` instead.)
    :param key_bindings: a :class:`.KeyBindings` object.
    :param get_cursor_position: A callable that returns the cursor position as
        a `Point` instance.
    """

    def __init__(self, text: AnyFormattedText='', style: str='', focusable: FilterOrBool=False, key_bindings: KeyBindingsBase | None=None, show_cursor: bool=True, modal: bool=False, get_cursor_position: Callable[[], Point | None] | None=None) -> None:
        self.text = text
        self.style = style
        self.focusable = to_filter(focusable)
        self.key_bindings = key_bindings
        self.show_cursor = show_cursor
        self.modal = modal
        self.get_cursor_position = get_cursor_position
        self._content_cache: SimpleCache[Hashable, UIContent] = SimpleCache(maxsize=18)
        self._fragment_cache: SimpleCache[int, StyleAndTextTuples] = SimpleCache(maxsize=1)
        self._fragments: StyleAndTextTuples | None = None

    def reset(self) -> None:
        self._fragments = None

    def is_focusable(self) -> bool:
        return self.focusable()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.text!r})'

    def _get_formatted_text_cached(self) -> StyleAndTextTuples:
        """
        Get fragments, but only retrieve fragments once during one render run.
        (This function is called several times during one rendering, because
        we also need those for calculating the dimensions.)
        """
        return self._fragment_cache.get(get_app().render_counter, lambda: to_formatted_text(self.text, self.style))

    def preferred_width(self, max_available_width: int) -> int:
        """
        Return the preferred width for this control.
        That is the width of the longest line.
        """
        text = fragment_list_to_text(self._get_formatted_text_cached())
        line_lengths = [get_cwidth(l) for l in text.split('\n')]
        return max(line_lengths)

    def preferred_height(self, width: int, max_available_height: int, wrap_lines: bool, get_line_prefix: GetLinePrefixCallable | None) -> int | None:
        """
        Return the preferred height for this control.
        """
        content = self.create_content(width, None)
        if wrap_lines:
            height = 0
            for i in range(content.line_count):
                height += content.get_height_for_line(i, width, get_line_prefix)
                if height >= max_available_height:
                    return max_available_height
            return height
        else:
            return content.line_count

    def create_content(self, width: int, height: int | None) -> UIContent:
        fragments_with_mouse_handlers = self._get_formatted_text_cached()
        fragment_lines_with_mouse_handlers = list(split_lines(fragments_with_mouse_handlers))
        fragment_lines: list[StyleAndTextTuples] = [[(item[0], item[1]) for item in line] for line in fragment_lines_with_mouse_handlers]
        self._fragments = fragments_with_mouse_handlers

        def get_cursor_position(fragment: str='[SetCursorPosition]') -> Point | None:
            for y, line in enumerate(fragment_lines):
                x = 0
                for style_str, text, *_ in line:
                    if fragment in style_str:
                        return Point(x=x, y=y)
                    x += len(text)
            return None

        def get_menu_position() -> Point | None:
            return get_cursor_position('[SetMenuPosition]')
        cursor_position = (self.get_cursor_position or get_cursor_position)()
        key = (tuple(fragments_with_mouse_handlers), width, cursor_position)

        def get_content() -> UIContent:
            return UIContent(get_line=lambda i: fragment_lines[i], line_count=len(fragment_lines), show_cursor=self.show_cursor, cursor_position=cursor_position, menu_position=get_menu_position())
        return self._content_cache.get(key, get_content)

    def mouse_handler(self, mouse_event: MouseEvent) -> NotImplementedOrNone:
        """
        Handle mouse events.

        (When the fragment list contained mouse handlers and the user clicked on
        on any of these, the matching handler is called. This handler can still
        return `NotImplemented` in case we want the
        :class:`~prompt_toolkit.layout.Window` to handle this particular
        event.)
        """
        if self._fragments:
            fragments_for_line = list(split_lines(self._fragments))
            try:
                fragments = fragments_for_line[mouse_event.position.y]
            except IndexError:
                return NotImplemented
            else:
                xpos = mouse_event.position.x
                count = 0
                for item in fragments:
                    count += len(item[1])
                    if count > xpos:
                        if len(item) >= 3:
                            handler = item[2]
                            return handler(mouse_event)
                        else:
                            break
        return NotImplemented

    def is_modal(self) -> bool:
        return self.modal

    def get_key_bindings(self) -> KeyBindingsBase | None:
        return self.key_bindings