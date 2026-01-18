from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class UndirectedFunctionApplication(UndirectedBinaryCombinator):
    """
    Class representing function application.
    Implements rules of the form:
    X/Y Y -> X (>)
    And the corresponding backwards application rule
    """

    def can_combine(self, function, argument):
        if not function.is_function():
            return False
        return not function.arg().can_unify(argument) is None

    def combine(self, function, argument):
        if not function.is_function():
            return
        subs = function.arg().can_unify(argument)
        if subs is None:
            return
        yield function.res().substitute(subs)

    def __str__(self):
        return ''