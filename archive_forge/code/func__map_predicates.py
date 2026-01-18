from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def _map_predicates(self, expression, predDict):
    if isinstance(expression, ApplicationExpression):
        func, args = expression.uncurry()
        if isinstance(func, AbstractVariableExpression):
            predDict[func].append_sig(tuple(args))
    elif isinstance(expression, AndExpression):
        self._map_predicates(expression.first, predDict)
        self._map_predicates(expression.second, predDict)
    elif isinstance(expression, AllExpression):
        sig = [expression.variable]
        term = expression.term
        while isinstance(term, AllExpression):
            sig.append(term.variable)
            term = term.term
        if isinstance(term, ImpExpression):
            if isinstance(term.first, ApplicationExpression) and isinstance(term.second, ApplicationExpression):
                func1, args1 = term.first.uncurry()
                func2, args2 = term.second.uncurry()
                if isinstance(func1, AbstractVariableExpression) and isinstance(func2, AbstractVariableExpression) and (sig == [v.variable for v in args1]) and (sig == [v.variable for v in args2]):
                    predDict[func2].append_prop((tuple(sig), term.first))
                    predDict[func1].validate_sig_len(sig)