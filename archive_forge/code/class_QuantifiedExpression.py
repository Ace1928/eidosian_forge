import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class QuantifiedExpression(VariableBinderExpression):

    @property
    def type(self):
        return TRUTH_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)
        if signature is None:
            signature = defaultdict(list)
        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.term._set_type(TRUTH_TYPE, signature)

    def __str__(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        return self.getQuantifier() + ' ' + ' '.join(('%s' % v for v in variables)) + Tokens.DOT + '%s' % term