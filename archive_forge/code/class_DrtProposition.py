import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtProposition(DrtExpression, Expression):

    def __init__(self, variable, drs):
        self.variable = variable
        self.drs = drs

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        if self.variable == variable:
            assert isinstance(expression, DrtAbstractVariableExpression), 'Can only replace a proposition label with a variable'
            return DrtProposition(expression.variable, self.drs.replace(variable, expression, replace_bound, alpha_convert))
        else:
            return DrtProposition(self.variable, self.drs.replace(variable, expression, replace_bound, alpha_convert))

    def eliminate_equality(self):
        return DrtProposition(self.variable, self.drs.eliminate_equality())

    def get_refs(self, recursive=False):
        return self.drs.get_refs(True) if recursive else []

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.variable == other.variable and (self.drs == other.drs)

    def __ne__(self, other):
        return not self == other
    __hash__ = Expression.__hash__

    def fol(self):
        return self.drs.fol()

    def _pretty(self):
        drs_s = self.drs._pretty()
        blank = ' ' * len('%s' % self.variable)
        return [blank + ' ' + line for line in drs_s[:1]] + ['%s' % self.variable + ':' + line for line in drs_s[1:2]] + [blank + ' ' + line for line in drs_s[2:]]

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.drs)])

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        return combinator(self.variable, function(self.drs))

    def __str__(self):
        return f'prop({self.variable}, {self.drs})'