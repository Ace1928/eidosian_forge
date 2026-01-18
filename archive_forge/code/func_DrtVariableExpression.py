import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def DrtVariableExpression(variable):
    """
    This is a factory method that instantiates and returns a subtype of
    ``DrtAbstractVariableExpression`` appropriate for the given variable.
    """
    if is_indvar(variable.name):
        return DrtIndividualVariableExpression(variable)
    elif is_funcvar(variable.name):
        return DrtFunctionVariableExpression(variable)
    elif is_eventvar(variable.name):
        return DrtEventVariableExpression(variable)
    else:
        return DrtConstantExpression(variable)