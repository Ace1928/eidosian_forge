import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_DrtProposition(self, expression, command, x, y):
    right = command(expression.variable, x, y)[0]
    right, bottom = self._handle(expression.term, command, right, y)
    return (right, bottom)