from abc import ABCMeta, abstractmethod
from functools import total_ordering
from nltk.internals import raise_unorderable_types
class PrimitiveCategory(AbstractCCGCategory):
    """
    Class representing primitive categories.
    Takes a string representation of the category, and a
    list of strings specifying the morphological subcategories.
    """

    def __init__(self, categ, restrictions=[]):
        self._categ = categ
        self._restrs = restrictions
        self._comparison_key = (categ, tuple(restrictions))

    def is_primitive(self):
        return True

    def is_function(self):
        return False

    def is_var(self):
        return False

    def restrs(self):
        return self._restrs

    def categ(self):
        return self._categ

    def substitute(self, subs):
        return self

    def can_unify(self, other):
        if not other.is_primitive():
            return None
        if other.is_var():
            return [(other, self)]
        if other.categ() == self.categ():
            for restr in self._restrs:
                if restr not in other.restrs():
                    return None
            return []
        return None

    def __str__(self):
        if self._restrs == []:
            return '%s' % self._categ
        restrictions = '[%s]' % ','.join((repr(r) for r in self._restrs))
        return f'{self._categ}{restrictions}'