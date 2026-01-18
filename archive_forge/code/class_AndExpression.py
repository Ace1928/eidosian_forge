import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class AndExpression(BooleanExpression):
    """This class represents conjunctions"""

    def getOp(self):
        return Tokens.AND

    def _str_subex(self, subex):
        s = '%s' % subex
        if isinstance(subex, AndExpression):
            return s[1:-1]
        return s