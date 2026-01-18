import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_VariableExpression(self, expression, command, x, y):
    return command('%s' % expression, x, y)