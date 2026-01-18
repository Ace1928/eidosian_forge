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
class UIControl(with_metaclass(ABCMeta, object)):
    """
    Base class for all user interface controls.
    """

    def reset(self):
        pass

    def preferred_width(self, cli, max_available_width):
        return None

    def preferred_height(self, cli, width, max_available_height, wrap_lines):
        return None

    def has_focus(self, cli):
        """
        Return ``True`` when this user control has the focus.

        If so, the cursor will be displayed according to the cursor position
        reported by :meth:`.UIControl.create_content`. If the created content
        has the property ``show_cursor=False``, the cursor will be hidden from
        the output.
        """
        return False

    @abstractmethod
    def create_content(self, cli, width, height):
        """
        Generate the content for this user control.

        Returns a :class:`.UIContent` instance.
        """

    def mouse_handler(self, cli, mouse_event):
        """
        Handle mouse events.

        When `NotImplemented` is returned, it means that the given event is not
        handled by the `UIControl` itself. The `Window` or key bindings can
        decide to handle this event as scrolling or changing focus.

        :param cli: `CommandLineInterface` instance.
        :param mouse_event: `MouseEvent` instance.
        """
        return NotImplemented

    def move_cursor_down(self, cli):
        """
        Request to move the cursor down.
        This happens when scrolling down and the cursor is completely at the
        top.
        """

    def move_cursor_up(self, cli):
        """
        Request to move the cursor up.
        """