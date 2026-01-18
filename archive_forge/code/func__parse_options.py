from __future__ import print_function
import sys
import getopt
from textwrap import dedent
from pygments import __version__, highlight
from pygments.util import ClassNotFound, OptionError, docstring_headline, \
from pygments.lexers import get_all_lexers, get_lexer_by_name, guess_lexer, \
from pygments.lexers.special import TextLexer
from pygments.formatters.latex import LatexEmbeddedLexer, LatexFormatter
from pygments.formatters import get_all_formatters, get_formatter_by_name, \
from pygments.formatters.terminal import TerminalFormatter
from pygments.filters import get_all_filters, find_filter_class
from pygments.styles import get_all_styles, get_style_by_name
def _parse_options(o_strs):
    opts = {}
    if not o_strs:
        return opts
    for o_str in o_strs:
        if not o_str.strip():
            continue
        o_args = o_str.split(',')
        for o_arg in o_args:
            o_arg = o_arg.strip()
            try:
                o_key, o_val = o_arg.split('=', 1)
                o_key = o_key.strip()
                o_val = o_val.strip()
            except ValueError:
                opts[o_arg] = True
            else:
                opts[o_key] = o_val
    return opts