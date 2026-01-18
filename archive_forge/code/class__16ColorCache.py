from __future__ import unicode_literals
from prompt_toolkit.filters import to_simple_filter, Condition
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from six.moves import range
import array
import errno
import os
import six
class _16ColorCache(dict):
    """
    Cache which maps (r, g, b) tuples to 16 ansi colors.

    :param bg: Cache for background colors, instead of foreground.
    """

    def __init__(self, bg=False):
        assert isinstance(bg, bool)
        self.bg = bg

    def get_code(self, value, exclude=()):
        """
        Return a (ansi_code, ansi_name) tuple. (E.g. ``(44, 'ansiblue')``.) for
        a given (r,g,b) value.
        """
        key = (value, exclude)
        if key not in self:
            self[key] = self._get(value, exclude)
        return self[key]

    def _get(self, value, exclude=()):
        r, g, b = value
        match = _get_closest_ansi_color(r, g, b, exclude=exclude)
        if self.bg:
            code = BG_ANSI_COLORS[match]
        else:
            code = FG_ANSI_COLORS[match]
        self[value] = code
        return (code, match)