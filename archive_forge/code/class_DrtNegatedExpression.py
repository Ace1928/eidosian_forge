import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtNegatedExpression(DrtExpression, NegatedExpression):

    def fol(self):
        return NegatedExpression(self.term.fol())

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return self.term.get_refs(recursive)

    def _pretty(self):
        term_lines = self.term._pretty()
        return ['    ' + line for line in term_lines[:2]] + ['__  ' + line for line in term_lines[2:3]] + ['  | ' + line for line in term_lines[3:4]] + ['    ' + line for line in term_lines[4:]]