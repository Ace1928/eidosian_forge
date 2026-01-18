import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
class LatexLexer(RegexpLexer, ABC):
    """A very simple lexer for tex/latex."""
    tokens = [('control_symbol_x2', '[\\\\][\\\\]|[\\\\]%'), ('comment', '%[^\\n]*'), ('control_word', '[\\\\][a-zA-Z]+'), ('control_symbol', '[\\\\][~\'"` =^!.]'), ('control_symbol_x', '[\\\\][^a-zA-Z]'), ('parameter', '\\#[0-9]|\\#'), ('space', ' |\\t'), ('newline', '\\n'), ('mathshift', '[$][$]|[$]'), ('chars', "---|--|-|[`][`]|['][']|[?][`]|[!][`]|(?![ %#$\\n\\t\\\\])."), ('unknown', '.')]
    'List of token names, and the regular expressions they match.'