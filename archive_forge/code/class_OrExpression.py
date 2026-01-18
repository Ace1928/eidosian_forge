import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class OrExpression(BooleanExpression):
    """This class represents disjunctions"""

    def getOp(self):
        return Tokens.OR

    def _str_subex(self, subex):
        s = '%s' % subex
        if isinstance(subex, OrExpression):
            return s[1:-1]
        return s