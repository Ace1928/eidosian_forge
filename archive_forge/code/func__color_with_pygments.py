import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _color_with_pygments(self, codeblock, lexer, **formatter_opts):
    import pygments
    import pygments.formatters

    class HtmlCodeFormatter(pygments.formatters.HtmlFormatter):

        def _wrap_code(self, inner):
            """A function for use in a Pygments Formatter which
                wraps in <code> tags.
                """
            yield (0, '<code>')
            for tup in inner:
                yield tup
            yield (0, '</code>')

        def _add_newline(self, inner):
            yield (0, '\n')
            yield from inner
            yield (0, '\n')

        def wrap(self, source, outfile=None):
            """Return the source with a code, pre, and div."""
            if outfile is None:
                return self._add_newline(self._wrap_pre(self._wrap_code(source)))
            else:
                return self._wrap_div(self._add_newline(self._wrap_pre(self._wrap_code(source))))
    formatter_opts.setdefault('cssclass', 'codehilite')
    formatter = HtmlCodeFormatter(**formatter_opts)
    return pygments.highlight(codeblock, lexer, formatter)