import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
class TerminalTrueColorFormatter(Terminal256Formatter):
    """
    Format tokens with ANSI color sequences, for output in a true-color
    terminal or console.  Like in `TerminalFormatter` color sequences
    are terminated at newlines, so that paging the output works correctly.

    .. versionadded:: 2.1

    Options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).
    """
    name = 'TerminalTrueColor'
    aliases = ['terminal16m', 'console16m', '16m']
    filenames = []

    def _build_color_table(self):
        pass

    def _color_tuple(self, color):
        try:
            rgb = int(str(color), 16)
        except ValueError:
            return None
        r = rgb >> 16 & 255
        g = rgb >> 8 & 255
        b = rgb & 255
        return (r, g, b)

    def _setup_styles(self):
        for ttype, ndef in self.style:
            escape = EscapeSequence()
            if ndef['color']:
                escape.fg = self._color_tuple(ndef['color'])
            if ndef['bgcolor']:
                escape.bg = self._color_tuple(ndef['bgcolor'])
            if self.usebold and ndef['bold']:
                escape.bold = True
            if self.useunderline and ndef['underline']:
                escape.underline = True
            self.style_string[str(ttype)] = (escape.true_color_string(), escape.reset_string())