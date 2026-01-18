import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
class Prover9Command(Prover9CommandParent, BaseProverCommand):
    """
    A ``ProverCommand`` specific to the ``Prover9`` prover.  It contains
    the a print_assumptions() method that is used to print the list
    of assumptions in multiple formats.
    """

    def __init__(self, goal=None, assumptions=None, timeout=60, prover=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        :param timeout: number of seconds before timeout; set to 0 for
            no timeout.
        :type timeout: int
        :param prover: a prover.  If not set, one will be created.
        :type prover: Prover9
        """
        if not assumptions:
            assumptions = []
        if prover is not None:
            assert isinstance(prover, Prover9)
        else:
            prover = Prover9(timeout)
        BaseProverCommand.__init__(self, prover, goal, assumptions)

    def decorate_proof(self, proof_string, simplify=True):
        """
        :see BaseProverCommand.decorate_proof()
        """
        if simplify:
            return self._prover._call_prooftrans(proof_string, ['striplabels'])[0].rstrip()
        else:
            return proof_string.rstrip()