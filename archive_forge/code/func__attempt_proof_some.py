from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_some(self, current, context, agenda, accessible_vars, atoms, debug):
    new_unique_variable = VariableExpression(unique_variable())
    agenda.put(current.term.replace(current.variable, new_unique_variable), context)
    agenda.mark_alls_fresh()
    return self._attempt_proof(agenda, accessible_vars | {new_unique_variable}, atoms, debug + 1)