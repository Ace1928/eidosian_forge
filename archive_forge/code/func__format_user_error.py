import os
from typing import Optional, Tuple, Union
from .util import NO_UTF8, color, supports_ansi
def _format_user_error(self, text, i, highlight):
    spacing = '  ' * i + ' >>>'
    if self.supports_ansi:
        spacing = color(spacing, fg=self.color_error)
    if highlight and self.supports_ansi:
        formatted_highlight = color(highlight, fg=self.color_highlight)
        text = text.replace(highlight, formatted_highlight)
    return '\n{}  {} {}'.format(self.indent, spacing, text)