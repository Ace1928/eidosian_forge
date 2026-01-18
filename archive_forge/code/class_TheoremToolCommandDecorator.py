import threading
import time
from abc import ABCMeta, abstractmethod
class TheoremToolCommandDecorator(TheoremToolCommand):
    """
    A base decorator for the ``ProverCommandDecorator`` and
    ``ModelBuilderCommandDecorator`` classes from which decorators can extend.
    """

    def __init__(self, command):
        """
        :param command: ``TheoremToolCommand`` to decorate
        """
        self._command = command
        self._result = None

    def assumptions(self):
        return self._command.assumptions()

    def goal(self):
        return self._command.goal()

    def add_assumptions(self, new_assumptions):
        self._command.add_assumptions(new_assumptions)
        self._result = None

    def retract_assumptions(self, retracted, debug=False):
        self._command.retract_assumptions(retracted, debug)
        self._result = None

    def print_assumptions(self):
        self._command.print_assumptions()