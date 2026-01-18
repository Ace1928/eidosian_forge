import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def _get_wrapped_text(self):
    """
        Return a copy of the text string with new lines added so that the text
        is wrapped relative to the parent figure (if `get_wrap` is True).
        """
    if not self.get_wrap():
        return self.get_text()
    if self.get_usetex():
        return self.get_text()
    line_width = self._get_wrap_line_width()
    wrapped_lines = []
    unwrapped_lines = self.get_text().split('\n')
    for unwrapped_line in unwrapped_lines:
        sub_words = unwrapped_line.split(' ')
        while len(sub_words) > 0:
            if len(sub_words) == 1:
                wrapped_lines.append(sub_words.pop(0))
                continue
            for i in range(2, len(sub_words) + 1):
                line = ' '.join(sub_words[:i])
                current_width = self._get_rendered_text_width(line)
                if current_width > line_width:
                    wrapped_lines.append(' '.join(sub_words[:i - 1]))
                    sub_words = sub_words[i - 1:]
                    break
                elif i == len(sub_words):
                    wrapped_lines.append(' '.join(sub_words[:i]))
                    sub_words = []
                    break
    return '\n'.join(wrapped_lines)