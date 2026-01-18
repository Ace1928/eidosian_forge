import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class IotaExpression(QuantifiedExpression):

    def getQuantifier(self):
        return Tokens.IOTA