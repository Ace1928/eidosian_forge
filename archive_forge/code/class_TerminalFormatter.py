import sys
from pygments.formatter import Formatter
from pygments.token import Keyword, Name, Comment, String, Error, \
from pygments.console import ansiformat
from pygments.util import get_choice_opt
class TerminalFormatter(Formatter):
    """
    Format tokens with ANSI color sequences, for output in a text console.
    Color sequences are terminated at newlines, so that paging the output
    works correctly.

    The `get_style_defs()` method doesn't do anything special since there is
    no support for common styles.

    Options accepted:

    `bg`
        Set to ``"light"`` or ``"dark"`` depending on the terminal's background
        (default: ``"light"``).

    `colorscheme`
        A dictionary mapping token types to (lightbg, darkbg) color names or
        ``None`` (default: ``None`` = use builtin colorscheme).

    `linenos`
        Set to ``True`` to have line numbers on the terminal output as well
        (default: ``False`` = no line numbers).
    """
    name = 'Terminal'
    aliases = ['terminal', 'console']
    filenames = []

    def __init__(self, **options):
        Formatter.__init__(self, **options)
        self.darkbg = get_choice_opt(options, 'bg', ['light', 'dark'], 'light') == 'dark'
        self.colorscheme = options.get('colorscheme', None) or TERMINAL_COLORS
        self.linenos = options.get('linenos', False)
        self._lineno = 0

    def format(self, tokensource, outfile):
        if not self.encoding and hasattr(outfile, 'encoding') and hasattr(outfile, 'isatty') and outfile.isatty() and (sys.version_info < (3,)):
            self.encoding = outfile.encoding
        return Formatter.format(self, tokensource, outfile)

    def _write_lineno(self, outfile):
        self._lineno += 1
        outfile.write('%s%04d: ' % (self._lineno != 1 and '\n' or '', self._lineno))

    def _get_color(self, ttype):
        colors = self.colorscheme.get(ttype)
        while colors is None:
            ttype = ttype.parent
            colors = self.colorscheme.get(ttype)
        return colors[self.darkbg]

    def format_unencoded(self, tokensource, outfile):
        if self.linenos:
            self._write_lineno(outfile)
        for ttype, value in tokensource:
            color = self._get_color(ttype)
            for line in value.splitlines(True):
                if color:
                    outfile.write(ansiformat(color, line.rstrip('\n')))
                else:
                    outfile.write(line.rstrip('\n'))
                if line.endswith('\n'):
                    if self.linenos:
                        self._write_lineno(outfile)
                    else:
                        outfile.write('\n')
        if self.linenos:
            outfile.write('\n')