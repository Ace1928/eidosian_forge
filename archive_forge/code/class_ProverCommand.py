import threading
import time
from abc import ABCMeta, abstractmethod
class ProverCommand(TheoremToolCommand):
    """
    This class holds a ``Prover``, a goal, and a list of assumptions.  When
    prove() is called, the ``Prover`` is executed with the goal and assumptions.
    """

    @abstractmethod
    def prove(self, verbose=False):
        """
        Perform the actual proof.
        """

    @abstractmethod
    def proof(self, simplify=True):
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """

    @abstractmethod
    def get_prover(self):
        """
        Return the prover object
        :return: ``Prover``
        """