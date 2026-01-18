import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class BasicType(Type):

    def __eq__(self, other):
        return isinstance(other, BasicType) and '%s' % self == '%s' % other

    def __ne__(self, other):
        return not self == other
    __hash__ = Type.__hash__

    def matches(self, other):
        return other == ANY_TYPE or self == other

    def resolve(self, other):
        if self.matches(other):
            return self
        else:
            return None