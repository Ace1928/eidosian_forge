from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_d_neg(self, current, context, agenda, accessible_vars, atoms, debug):
    agenda.put(current.term.term, context)
    return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)