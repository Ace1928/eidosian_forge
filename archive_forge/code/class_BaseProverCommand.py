import threading
import time
from abc import ABCMeta, abstractmethod
class BaseProverCommand(BaseTheoremToolCommand, ProverCommand):
    """
    This class holds a ``Prover``, a goal, and a list of assumptions.  When
    prove() is called, the ``Prover`` is executed with the goal and assumptions.
    """

    def __init__(self, prover, goal=None, assumptions=None):
        """
        :param prover: The theorem tool to execute with the assumptions
        :type prover: Prover
        :see: ``BaseTheoremToolCommand``
        """
        self._prover = prover
        'The theorem tool to execute with the assumptions'
        BaseTheoremToolCommand.__init__(self, goal, assumptions)
        self._proof = None

    def prove(self, verbose=False):
        """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
        if self._result is None:
            self._result, self._proof = self._prover._prove(self.goal(), self.assumptions(), verbose)
        return self._result

    def proof(self, simplify=True):
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
        if self._result is None:
            raise LookupError('You have to call prove() first to get a proof!')
        else:
            return self.decorate_proof(self._proof, simplify)

    def decorate_proof(self, proof_string, simplify=True):
        """
        Modify and return the proof string
        :param proof_string: str the proof to decorate
        :param simplify: bool simplify the proof?
        :return: str
        """
        return proof_string

    def get_prover(self):
        return self._prover