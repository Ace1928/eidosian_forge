from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def _attempt_proof_eq(self, current, context, agenda, accessible_vars, atoms, debug):
    agenda.put_atoms(atoms)
    agenda.replace_all(current.first, current.second)
    accessible_vars.discard(current.first)
    agenda.mark_neqs_fresh()
    return self._attempt_proof(agenda, accessible_vars, set(), debug + 1)