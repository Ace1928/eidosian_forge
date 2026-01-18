import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
@staticmethod
def _is_simple_character(c):
    return isinstance(c, Character) and c.positive and (not c.case_flags)