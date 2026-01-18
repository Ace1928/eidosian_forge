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
def _color_name_to_rgb(self, color):
    """ Turn 'ffffff', into (0xff, 0xff, 0xff). """
    try:
        rgb = int(color, 16)
    except ValueError:
        raise
    else:
        r = rgb >> 16 & 255
        g = rgb >> 8 & 255
        b = rgb & 255
        return (r, g, b)