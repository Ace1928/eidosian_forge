import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class LambdaExpression(VariableBinderExpression):

    @property
    def type(self):
        return ComplexType(self.term.findtype(self.variable), self.term.type)

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)
        if signature is None:
            signature = defaultdict(list)
        self.term._set_type(other_type.second, signature)
        if not self.type.resolve(other_type):
            raise TypeResolutionException(self, other_type)

    def __str__(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        return Tokens.LAMBDA + ' '.join(('%s' % v for v in variables)) + Tokens.DOT + '%s' % term