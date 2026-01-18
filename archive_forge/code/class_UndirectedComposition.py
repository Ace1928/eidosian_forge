from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class UndirectedComposition(UndirectedBinaryCombinator):
    """
    Functional composition (harmonic) combinator.
    Implements rules of the form
    X/Y Y/Z -> X/Z (B>)
    And the corresponding backwards and crossed variations.
    """

    def can_combine(self, function, argument):
        if not (function.is_function() and argument.is_function()):
            return False
        if function.dir().can_compose() and argument.dir().can_compose():
            return not function.arg().can_unify(argument.res()) is None
        return False

    def combine(self, function, argument):
        if not (function.is_function() and argument.is_function()):
            return
        if function.dir().can_compose() and argument.dir().can_compose():
            subs = function.arg().can_unify(argument.res())
            if subs is not None:
                yield FunctionalCategory(function.res().substitute(subs), argument.arg().substitute(subs), argument.dir())

    def __str__(self):
        return 'B'