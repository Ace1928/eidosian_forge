from __future__ import unicode_literals
import re
from pybtex.exceptions import PybtexError
from pybtex import py3compat
class TokenRequired(PybtexSyntaxError):

    def __init__(self, description, parser):
        message = u'{0} expected'.format(description)
        super(TokenRequired, self).__init__(message, parser)

    def get_context(self):
        context, lineno, colno = self.parser.get_error_context(self.error_context_info)
        if context is None:
            return ''
        if colno == 0:
            marker = '^^'
        else:
            marker = ' ' * (colno - 1) + '^^^'
        return '\n'.join((context, marker))