import re
from ..core.inputscanner import InputScanner
from ..core.token import Token
from ..core.tokenstream import TokenStream
from ..core.pattern import Pattern
from ..core.whitespacepattern import WhitespacePattern
def _readWhitespace(self):
    return self._patterns.whitespace.read()