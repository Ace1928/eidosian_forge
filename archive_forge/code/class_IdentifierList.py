from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
class IdentifierList(TokenList):
    """A list of :class:`~sqlparse.sql.Identifier`'s."""

    def get_identifiers(self):
        """Returns the identifiers.

        Whitespaces and punctuations are not included in this generator.
        """
        for token in self.tokens:
            if not (token.is_whitespace or token.match(T.Punctuation, ',')):
                yield token