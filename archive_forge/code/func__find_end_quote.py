from __future__ import (absolute_import, division, print_function)
import re
def _find_end_quote(identifier, quote_char):
    accumulate = 0
    while True:
        try:
            quote = identifier.index(quote_char)
        except ValueError:
            raise UnclosedQuoteError
        accumulate = accumulate + quote
        try:
            next_char = identifier[quote + 1]
        except IndexError:
            return accumulate
        if next_char == quote_char:
            try:
                identifier = identifier[quote + 2:]
                accumulate = accumulate + 2
            except IndexError:
                raise UnclosedQuoteError
        else:
            return accumulate