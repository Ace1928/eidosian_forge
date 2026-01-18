import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _format_syntax_error(self, stype):
    """Format SyntaxError exceptions (internal helper)."""
    filename_suffix = ''
    if self.lineno is not None:
        yield '  File "{}", line {}\n'.format(self.filename or '<string>', self.lineno)
    elif self.filename is not None:
        filename_suffix = ' ({})'.format(self.filename)
    text = self.text
    if text is not None:
        rtext = text.rstrip('\n')
        ltext = rtext.lstrip(' \n\x0c')
        spaces = len(rtext) - len(ltext)
        yield '    {}\n'.format(ltext)
        if self.offset is not None:
            offset = self.offset
            end_offset = self.end_offset if self.end_offset not in {None, 0} else offset
            if offset == end_offset or end_offset == -1:
                end_offset = offset + 1
            colno = offset - 1 - spaces
            end_colno = end_offset - 1 - spaces
            if colno >= 0:
                caretspace = (c if c.isspace() else ' ' for c in ltext[:colno])
                yield '    {}{}'.format(''.join(caretspace), '^' * (end_colno - colno) + '\n')
    msg = self.msg or '<no detail available>'
    yield '{}: {}{}\n'.format(stype, msg, filename_suffix)