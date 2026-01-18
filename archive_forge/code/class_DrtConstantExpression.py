import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtConstantExpression(DrtAbstractVariableExpression, ConstantExpression):
    pass