import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class EqualityExpression(BinaryExpression):
    """This class represents equality expressions like "(x = y)"."""

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)
        if signature is None:
            signature = defaultdict(list)
        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.first._set_type(ENTITY_TYPE, signature)
        self.second._set_type(ENTITY_TYPE, signature)

    def getOp(self):
        return Tokens.EQ