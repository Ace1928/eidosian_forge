import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
class ResolutionProver(Prover):
    ANSWER_KEY = 'ANSWER'
    _assume_false = True

    def _prove(self, goal=None, assumptions=None, verbose=False):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in the proof
        :type assumptions: list(sem.Expression)
        """
        if not assumptions:
            assumptions = []
        result = None
        try:
            clauses = []
            if goal:
                clauses.extend(clausify(-goal))
            for a in assumptions:
                clauses.extend(clausify(a))
            result, clauses = self._attempt_proof(clauses)
            if verbose:
                print(ResolutionProverCommand._decorate_clauses(clauses))
        except RuntimeError as e:
            if self._assume_false and str(e).startswith('maximum recursion depth exceeded'):
                result = False
                clauses = []
            elif verbose:
                print(e)
            else:
                raise e
        return (result, clauses)

    def _attempt_proof(self, clauses):
        tried = defaultdict(list)
        i = 0
        while i < len(clauses):
            if not clauses[i].is_tautology():
                if tried[i]:
                    j = tried[i][-1] + 1
                else:
                    j = i + 1
                while j < len(clauses):
                    if i != j and j and (not clauses[j].is_tautology()):
                        tried[i].append(j)
                        newclauses = clauses[i].unify(clauses[j])
                        if newclauses:
                            for newclause in newclauses:
                                newclause._parents = (i + 1, j + 1)
                                clauses.append(newclause)
                                if not len(newclause):
                                    return (True, clauses)
                            i = -1
                            break
                    j += 1
            i += 1
        return (False, clauses)