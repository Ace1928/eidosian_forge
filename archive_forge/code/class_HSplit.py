from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from .controls import UIControl, TokenListControl, UIContent
from .dimension import LayoutDimension, sum_layout_dimensions, max_layout_dimensions
from .margins import Margin
from .screen import Point, WritePosition, _CHAR_CACHE
from .utils import token_list_to_text, explode_tokens
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import to_cli_filter, ViInsertMode, EmacsInsertMode
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from prompt_toolkit.utils import take_using_weights, get_cwidth
class HSplit(Container):
    """
    Several layouts, one stacked above/under the other.

    :param children: List of child :class:`.Container` objects.
    :param window_too_small: A :class:`.Container` object that is displayed if
        there is not enough space for all the children. By default, this is a
        "Window too small" message.
    :param get_dimensions: (`None` or a callable that takes a
        `CommandLineInterface` and returns a list of `LayoutDimension`
        instances.) By default the dimensions are taken from the children and
        divided by the available space. However, when `get_dimensions` is specified,
        this is taken instead.
    :param report_dimensions_callback: When rendering, this function is called
        with the `CommandLineInterface` and the list of used dimensions. (As a
        list of integers.)
    """

    def __init__(self, children, window_too_small=None, get_dimensions=None, report_dimensions_callback=None):
        assert all((isinstance(c, Container) for c in children))
        assert window_too_small is None or isinstance(window_too_small, Container)
        assert get_dimensions is None or callable(get_dimensions)
        assert report_dimensions_callback is None or callable(report_dimensions_callback)
        self.children = children
        self.window_too_small = window_too_small or _window_too_small()
        self.get_dimensions = get_dimensions
        self.report_dimensions_callback = report_dimensions_callback

    def preferred_width(self, cli, max_available_width):
        if self.children:
            dimensions = [c.preferred_width(cli, max_available_width) for c in self.children]
            return max_layout_dimensions(dimensions)
        else:
            return LayoutDimension(0)

    def preferred_height(self, cli, width, max_available_height):
        dimensions = [c.preferred_height(cli, width, max_available_height) for c in self.children]
        return sum_layout_dimensions(dimensions)

    def reset(self):
        for c in self.children:
            c.reset()

    def write_to_screen(self, cli, screen, mouse_handlers, write_position):
        """
        Render the prompt to a `Screen` instance.

        :param screen: The :class:`~prompt_toolkit.layout.screen.Screen` class
            to which the output has to be written.
        """
        sizes = self._divide_heigths(cli, write_position)
        if self.report_dimensions_callback:
            self.report_dimensions_callback(cli, sizes)
        if sizes is None:
            self.window_too_small.write_to_screen(cli, screen, mouse_handlers, write_position)
        else:
            ypos = write_position.ypos
            xpos = write_position.xpos
            width = write_position.width
            for s, c in zip(sizes, self.children):
                c.write_to_screen(cli, screen, mouse_handlers, WritePosition(xpos, ypos, width, s))
                ypos += s

    def _divide_heigths(self, cli, write_position):
        """
        Return the heights for all rows.
        Or None when there is not enough space.
        """
        if not self.children:
            return []
        given_dimensions = self.get_dimensions(cli) if self.get_dimensions else None

        def get_dimension_for_child(c, index):
            if given_dimensions and given_dimensions[index] is not None:
                return given_dimensions[index]
            else:
                return c.preferred_height(cli, write_position.width, write_position.extended_height)
        dimensions = [get_dimension_for_child(c, index) for index, c in enumerate(self.children)]
        sum_dimensions = sum_layout_dimensions(dimensions)
        if sum_dimensions.min > write_position.extended_height:
            return
        sizes = [d.min for d in dimensions]
        child_generator = take_using_weights(items=list(range(len(dimensions))), weights=[d.weight for d in dimensions])
        i = next(child_generator)
        while sum(sizes) < min(write_position.extended_height, sum_dimensions.preferred):
            if sizes[i] < dimensions[i].preferred:
                sizes[i] += 1
            i = next(child_generator)
        if not any([cli.is_returning, cli.is_exiting, cli.is_aborting]):
            while sum(sizes) < min(write_position.height, sum_dimensions.max):
                if sizes[i] < dimensions[i].max:
                    sizes[i] += 1
                i = next(child_generator)
        return sizes

    def walk(self, cli):
        """ Walk through children. """
        yield self
        for c in self.children:
            for i in c.walk(cli):
                yield i