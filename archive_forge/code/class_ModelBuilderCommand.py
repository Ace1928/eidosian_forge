import threading
import time
from abc import ABCMeta, abstractmethod
class ModelBuilderCommand(TheoremToolCommand):
    """
    This class holds a ``ModelBuilder``, a goal, and a list of assumptions.
    When build_model() is called, the ``ModelBuilder`` is executed with the goal
    and assumptions.
    """

    @abstractmethod
    def build_model(self, verbose=False):
        """
        Perform the actual model building.
        :return: A model if one is generated; None otherwise.
        :rtype: sem.Valuation
        """

    @abstractmethod
    def model(self, format=None):
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """

    @abstractmethod
    def get_model_builder(self):
        """
        Return the model builder object
        :return: ``ModelBuilder``
        """