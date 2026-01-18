import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class PossibleAntecedents(list, DrtExpression, Expression):

    def free(self):
        """Set of free variables."""
        return set(self)

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        result = PossibleAntecedents()
        for item in self:
            if item == variable:
                self.append(expression)
            else:
                self.append(item)
        return result

    def _pretty(self):
        s = '%s' % self
        blank = ' ' * len(s)
        return [blank, blank, s]

    def __str__(self):
        return '[' + ','.join(('%s' % it for it in self)) + ']'