import threading
import time
from abc import ABCMeta, abstractmethod
class TheoremToolCommand(metaclass=ABCMeta):
    """
    This class holds a goal and a list of assumptions to be used in proving
    or model building.
    """

    @abstractmethod
    def add_assumptions(self, new_assumptions):
        """
        Add new assumptions to the assumption list.

        :param new_assumptions: new assumptions
        :type new_assumptions: list(sem.Expression)
        """

    @abstractmethod
    def retract_assumptions(self, retracted, debug=False):
        """
        Retract assumptions from the assumption list.

        :param debug: If True, give warning when ``retracted`` is not present on
            assumptions list.
        :type debug: bool
        :param retracted: assumptions to be retracted
        :type retracted: list(sem.Expression)
        """

    @abstractmethod
    def assumptions(self):
        """
        List the current assumptions.

        :return: list of ``Expression``
        """

    @abstractmethod
    def goal(self):
        """
        Return the goal

        :return: ``Expression``
        """

    @abstractmethod
    def print_assumptions(self):
        """
        Print the list of the current assumptions.
        """