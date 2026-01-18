from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
class NameFormatParser(Scanner):
    LBRACE = Literal(u'{')
    RBRACE = Literal(u'}')
    TEXT = Pattern('[^{}]+', 'text')
    NON_LETTERS = Pattern('[^{}\\w]|\\d+', 'non-letter characters', flags=re.IGNORECASE | re.UNICODE)
    FORMAT_CHARS = Pattern('[^\\W\\d_]+', 'format chars', flags=re.IGNORECASE | re.UNICODE)
    lineno = None

    def parse(self):
        while True:
            try:
                result = self.parse_toplevel()
                yield result
            except EOFError:
                break

    def parse_toplevel(self):
        token = self.required([self.TEXT, self.LBRACE, self.RBRACE], allow_eof=True)
        if token.pattern is self.TEXT:
            return Text(token.value)
        elif token.pattern is self.LBRACE:
            return NamePart(self.parse_name_part())
        elif token.pattern is self.RBRACE:
            raise UnbalancedBraceError(self)

    def parse_braced_string(self):
        while True:
            try:
                token = self.required([self.TEXT, self.RBRACE, self.LBRACE])
            except PrematureEOF:
                raise UnbalancedBraceError(self)
            if token.pattern is self.TEXT:
                yield token.value
            elif token.pattern is self.RBRACE:
                break
            elif token.pattern is self.LBRACE:
                yield u'{{{0}}}'.format(''.join(self.parse_braced_string()))
            else:
                raise ValueError(token)

    def parse_name_part(self):
        verbatim_prefix = []
        format_chars = None
        verbatim_postfix = []
        verbatim = verbatim_prefix
        delimiter = None

        def check_format_chars(value):
            value = value.lower()
            if format_chars is not None or len(value) not in [1, 2] or value[0] != value[-1] or (value[0] not in 'flvj'):
                raise PybtexSyntaxError(u'name format string "{0}" has illegal brace-level-1 letters: {1}'.format(self.text, token.value), self)
        while True:
            try:
                token = self.required([self.LBRACE, self.NON_LETTERS, self.FORMAT_CHARS, self.RBRACE])
            except PrematureEOF:
                raise UnbalancedBraceError(self)
            if token.pattern is self.LBRACE:
                verbatim.append(u'{{{0}}}'.format(''.join(self.parse_braced_string())))
            elif token.pattern is self.FORMAT_CHARS:
                check_format_chars(token.value)
                format_chars = token.value
                verbatim = verbatim_postfix
                if self.optional([self.LBRACE]):
                    delimiter = ''.join(self.parse_braced_string())
            elif token.pattern is self.NON_LETTERS:
                verbatim.append(token.value)
            elif token.pattern is self.RBRACE:
                return (''.join(verbatim_prefix), format_chars, delimiter, ''.join(verbatim_postfix))
            else:
                raise ValueError(token)

    def eat_whitespace(self):
        pass