from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
class Margin(with_metaclass(ABCMeta, object)):
    """
    Base interface for a margin.
    """

    @abstractmethod
    def get_width(self, cli, get_ui_content):
        """
        Return the width that this margin is going to consume.

        :param cli: :class:`.CommandLineInterface` instance.
        :param get_ui_content: Callable that asks the user control to create
            a :class:`.UIContent` instance. This can be used for instance to
            obtain the number of lines.
        """
        return 0

    @abstractmethod
    def create_margin(self, cli, window_render_info, width, height):
        """
        Creates a margin.
        This should return a list of (Token, text) tuples.

        :param cli: :class:`.CommandLineInterface` instance.
        :param window_render_info:
            :class:`~prompt_toolkit.layout.containers.WindowRenderInfo`
            instance, generated after rendering and copying the visible part of
            the :class:`~prompt_toolkit.layout.controls.UIControl` into the
            :class:`~prompt_toolkit.layout.containers.Window`.
        :param width: The width that's available for this margin. (As reported
            by :meth:`.get_width`.)
        :param height: The height that's available for this margin. (The height
            of the :class:`~prompt_toolkit.layout.containers.Window`.)
        """
        return []