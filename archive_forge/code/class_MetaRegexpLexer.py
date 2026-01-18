import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
class MetaRegexpLexer(ABCMeta):
    """Metaclass for :class:`RegexpLexer`. Compiles tokens into a
    regular expression.
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        regexp_string = '|'.join(('(?P<' + name + '>' + regexp + ')' for name, regexp in getattr(cls, 'tokens', [])))
        cls.regexp = re.compile(regexp_string, re.DOTALL)