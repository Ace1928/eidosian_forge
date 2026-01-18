import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtAbstractVariableExpression(DrtExpression, AbstractVariableExpression):

    def fol(self):
        return self

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return []

    def _pretty(self):
        s = '%s' % self
        blank = ' ' * len(s)
        return [blank, blank, s, blank]

    def eliminate_equality(self):
        return self