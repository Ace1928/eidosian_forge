from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
class ClosedWorldProver(ProverCommandDecorator):
    """
    This is a prover decorator that completes predicates before proving.

    If the assumptions contain "P(A)", then "all x.(P(x) -> (x=A))" is the completion of "P".
    If the assumptions contain "all x.(ostrich(x) -> bird(x))", then "all x.(bird(x) -> ostrich(x))" is the completion of "bird".
    If the assumptions don't contain anything that are "P", then "all x.-P(x)" is the completion of "P".

    walk(Socrates)
    Socrates != Bill
    + all x.(walk(x) -> (x=Socrates))
    ----------------
    -walk(Bill)

    see(Socrates, John)
    see(John, Mary)
    Socrates != John
    John != Mary
    + all x.all y.(see(x,y) -> ((x=Socrates & y=John) | (x=John & y=Mary)))
    ----------------
    -see(Socrates, Mary)

    all x.(ostrich(x) -> bird(x))
    bird(Tweety)
    -ostrich(Sam)
    Sam != Tweety
    + all x.(bird(x) -> (ostrich(x) | x=Tweety))
    + all x.-ostrich(x)
    -------------------
    -bird(Sam)
    """

    def assumptions(self):
        assumptions = self._command.assumptions()
        predicates = self._make_predicate_dict(assumptions)
        new_assumptions = []
        for p in predicates:
            predHolder = predicates[p]
            new_sig = self._make_unique_signature(predHolder)
            new_sig_exs = [VariableExpression(v) for v in new_sig]
            disjuncts = []
            for sig in predHolder.signatures:
                equality_exs = []
                for v1, v2 in zip(new_sig_exs, sig):
                    equality_exs.append(EqualityExpression(v1, v2))
                disjuncts.append(reduce(lambda x, y: x & y, equality_exs))
            for prop in predHolder.properties:
                bindings = {}
                for v1, v2 in zip(new_sig_exs, prop[0]):
                    bindings[v2] = v1
                disjuncts.append(prop[1].substitute_bindings(bindings))
            if disjuncts:
                antecedent = self._make_antecedent(p, new_sig)
                consequent = reduce(lambda x, y: x | y, disjuncts)
                accum = ImpExpression(antecedent, consequent)
            else:
                accum = NegatedExpression(self._make_antecedent(p, new_sig))
            for new_sig_var in new_sig[::-1]:
                accum = AllExpression(new_sig_var, accum)
            new_assumptions.append(accum)
        return assumptions + new_assumptions

    def _make_unique_signature(self, predHolder):
        """
        This method figures out how many arguments the predicate takes and
        returns a tuple containing that number of unique variables.
        """
        return tuple((unique_variable() for i in range(predHolder.signature_len)))

    def _make_antecedent(self, predicate, signature):
        """
        Return an application expression with 'predicate' as the predicate
        and 'signature' as the list of arguments.
        """
        antecedent = predicate
        for v in signature:
            antecedent = antecedent(VariableExpression(v))
        return antecedent

    def _make_predicate_dict(self, assumptions):
        """
        Create a dictionary of predicates from the assumptions.

        :param assumptions: a list of ``Expression``s
        :return: dict mapping ``AbstractVariableExpression`` to ``PredHolder``
        """
        predicates = defaultdict(PredHolder)
        for a in assumptions:
            self._map_predicates(a, predicates)
        return predicates

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