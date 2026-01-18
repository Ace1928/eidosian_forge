import re
from typing import List, Optional, Tuple
class _Word:

    def process(self, next_char, context):
        if _whitespace_match(next_char):
            return None
        elif next_char in context.allowed_quote_chars:
            return _Quotes(next_char, self)
        elif next_char == '\\':
            return _Backslash(self)
        else:
            context.token.append(next_char)
            return self