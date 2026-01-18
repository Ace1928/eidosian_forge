from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class DirectedBinaryCombinator(metaclass=ABCMeta):
    """
    Wrapper for the undirected binary combinator.
    It takes left and right categories, and decides which is to be
    the function, and which the argument.
    It then decides whether or not they can be combined.
    """

    @abstractmethod
    def can_combine(self, left, right):
        pass

    @abstractmethod
    def combine(self, left, right):
        pass