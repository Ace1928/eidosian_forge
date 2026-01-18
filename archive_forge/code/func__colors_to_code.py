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
def _colors_to_code(self, fg_color, bg_color):
    """ Return a tuple with the vt100 values  that represent this color. """
    fg_ansi = [()]

    def get(color, bg):
        table = BG_ANSI_COLORS if bg else FG_ANSI_COLORS
        if color is None:
            return ()
        elif color in table:
            return (table[color],)
        else:
            try:
                rgb = self._color_name_to_rgb(color)
            except ValueError:
                return ()
            if self.ansi_colors_only():
                if bg:
                    if fg_color != bg_color:
                        exclude = (fg_ansi[0],)
                    else:
                        exclude = ()
                    code, name = _16_bg_colors.get_code(rgb, exclude=exclude)
                    return (code,)
                else:
                    code, name = _16_fg_colors.get_code(rgb)
                    fg_ansi[0] = name
                    return (code,)
            elif self.true_color:
                r, g, b = rgb
                return (48 if bg else 38, 2, r, g, b)
            else:
                return (48 if bg else 38, 5, _256_colors[rgb])
    result = []
    result.extend(get(fg_color, False))
    result.extend(get(bg_color, True))
    return map(six.text_type, result)