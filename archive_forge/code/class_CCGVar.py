from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
class CCGVar(AbstractCCGCategory):
    """
    Class representing a variable CCG category.
    Used for conjunctions (and possibly type-raising, if implemented as a
    unary rule).
    """
    _maxID = 0

    def __init__(self, prim_only=False):
        """Initialize a variable (selects a new identifier)

        :param prim_only: a boolean that determines whether the variable is
                          restricted to primitives
        :type prim_only: bool
        """
        self._id = self.new_id()
        self._prim_only = prim_only
        self._comparison_key = self._id

    @classmethod
    def new_id(cls):
        """
        A class method allowing generation of unique variable identifiers.
        """
        cls._maxID = cls._maxID + 1
        return cls._maxID - 1

    @classmethod
    def reset_id(cls):
        cls._maxID = 0

    def is_primitive(self):
        return False

    def is_function(self):
        return False

    def is_var(self):
        return True

    def substitute(self, substitutions):
        """If there is a substitution corresponding to this variable,
        return the substituted category.
        """
        for var, cat in substitutions:
            if var == self:
                return cat
        return self

    def can_unify(self, other):
        """If the variable can be replaced with other
        a substitution is returned.
        """
        if other.is_primitive() or not self._prim_only:
            return [(self, other)]
        return None

    def id(self):
        return self._id

    def __str__(self):
        return '_var' + str(self._id)