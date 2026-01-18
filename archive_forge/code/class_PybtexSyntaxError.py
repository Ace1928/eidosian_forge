from __future__ import unicode_literals
import re
from pybtex.exceptions import PybtexError
from pybtex import py3compat
@py3compat.python_2_unicode_compatible
class PybtexSyntaxError(PybtexError):
    error_type = 'syntax error'

    def __init__(self, message, parser):
        super(PybtexSyntaxError, self).__init__(message, filename=parser.filename)
        self.lineno = parser.lineno
        self.parser = parser
        self.error_context_info = parser.get_error_context_info()

    def __str__(self):
        base_message = super(PybtexSyntaxError, self).__str__()
        pos = u' in line {0}'.format(self.lineno) if self.lineno is not None else ''
        return u'{error_type}{pos}: {message}'.format(error_type=self.error_type, pos=pos, message=base_message)