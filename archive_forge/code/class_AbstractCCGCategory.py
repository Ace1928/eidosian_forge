from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
@total_ordering
class AbstractCCGCategory(metaclass=ABCMeta):
    """
    Interface for categories in combinatory grammars.
    """

    @abstractmethod
    def is_primitive(self):
        """
        Returns true if the category is primitive.
        """

    @abstractmethod
    def is_function(self):
        """
        Returns true if the category is a function application.
        """

    @abstractmethod
    def is_var(self):
        """
        Returns true if the category is a variable.
        """

    @abstractmethod
    def substitute(self, substitutions):
        """
        Takes a set of (var, category) substitutions, and replaces every
        occurrence of the variable with the corresponding category.
        """

    @abstractmethod
    def can_unify(self, other):
        """
        Determines whether two categories can be unified.
         - Returns None if they cannot be unified
         - Returns a list of necessary substitutions if they can.
        """

    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._comparison_key == other._comparison_key

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, AbstractCCGCategory):
            raise_unorderable_types('<', self, other)
        if self.__class__ is other.__class__:
            return self._comparison_key < other._comparison_key
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._comparison_key)
            return self._hash