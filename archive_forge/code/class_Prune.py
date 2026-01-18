import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Prune(ZeroWidthBase):
    _op_name = 'PRUNE'

    def _compile(self, reverse, fuzzy):
        return [(OP.PRUNE,)]