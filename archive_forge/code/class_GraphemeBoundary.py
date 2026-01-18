import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class GraphemeBoundary:

    def compile(self, reverse, fuzzy):
        return [(OP.GRAPHEME_BOUNDARY, 1)]