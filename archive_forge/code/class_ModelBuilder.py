import threading
import time
from abc import ABCMeta, abstractmethod
class ModelBuilder(metaclass=ABCMeta):
    """
    Interface for trying to build a model of set of formulas.
    Open formulas are assumed to be universally quantified.
    Both the goal and the assumptions are constrained to be formulas
    of ``logic.Expression``.
    """

    def build_model(self, goal=None, assumptions=None, verbose=False):
        """
        Perform the actual model building.
        :return: Whether a model was generated
        :rtype: bool
        """
        return self._build_model(goal, assumptions, verbose)[0]

    @abstractmethod
    def _build_model(self, goal=None, assumptions=None, verbose=False):
        """
        Perform the actual model building.
        :return: Whether a model was generated, and the model itself
        :rtype: tuple(bool, sem.Valuation)
        """