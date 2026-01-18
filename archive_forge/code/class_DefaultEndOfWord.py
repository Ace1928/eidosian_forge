import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class DefaultEndOfWord(ZeroWidthBase):
    _opcode = OP.DEFAULT_END_OF_WORD
    _op_name = 'DEFAULT_END_OF_WORD'