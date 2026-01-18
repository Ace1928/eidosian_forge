from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_n_eq(self, current, context, agenda, accessible_vars, atoms, debug):
    if current.term.first == current.term.second:
        debug.line('CLOSED', 1)
        return True
    agenda[Categories.N_EQ].add((current, context))
    current._exhausted = True
    return self._attempt_proof(agenda, accessible_vars | {current.term.first, current.term.second}, atoms, debug + 1)