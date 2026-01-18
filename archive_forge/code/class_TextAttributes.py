from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class TextAttributes(object):
    """Attributes to use to style text with."""

    def __init__(self, format_str=None, color=None, attrs=None):
        """Defines a set of attributes for a piece of text.

    Args:
      format_str: (str), string that will be used to format the text
        with. For example '[{}]', to enclose text in brackets.
      color: (Colors), the color the text should be formatted with.
      attrs: (Attrs), the attributes to apply to text.
    """
        self._format_str = format_str
        self._color = color
        self._attrs = attrs or []

    @property
    def format_str(self):
        return self._format_str

    @property
    def color(self):
        return self._color

    @property
    def attrs(self):
        return self._attrs