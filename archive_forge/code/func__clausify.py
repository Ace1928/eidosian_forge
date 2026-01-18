import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _clausify(expression):
    """
    :param expression: a skolemized expression in CNF
    """
    if isinstance(expression, AndExpression):
        return _clausify(expression.first) + _clausify(expression.second)
    elif isinstance(expression, OrExpression):
        first = _clausify(expression.first)
        second = _clausify(expression.second)
        assert len(first) == 1
        assert len(second) == 1
        return [first[0] + second[0]]
    elif isinstance(expression, EqualityExpression):
        return [Clause([expression])]
    elif isinstance(expression, ApplicationExpression):
        return [Clause([expression])]
    elif isinstance(expression, NegatedExpression):
        if isinstance(expression.term, ApplicationExpression):
            return [Clause([expression])]
        elif isinstance(expression.term, EqualityExpression):
            return [Clause([expression])]
    raise ProverParseError()