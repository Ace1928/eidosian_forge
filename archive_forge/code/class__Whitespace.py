import re
from typing import List, Optional, Tuple
class _Whitespace:

    def process(self, next_char, context):
        if _whitespace_match(next_char):
            if len(context.token) > 0:
                return None
            else:
                return self
        elif next_char in context.allowed_quote_chars:
            context.quoted = True
            return _Quotes(next_char, self)
        elif next_char == '\\':
            return _Backslash(self)
        else:
            context.token.append(next_char)
            return _Word()