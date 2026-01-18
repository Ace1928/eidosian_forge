from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
class UndirectedTypeRaise(UndirectedBinaryCombinator):
    """
    Undirected combinator for type raising.
    """

    def can_combine(self, function, arg):
        if not (arg.is_function() and arg.res().is_function()):
            return False
        arg = innermostFunction(arg)
        subs = left.can_unify(arg_categ.arg())
        if subs is not None:
            return True
        return False

    def combine(self, function, arg):
        if not (function.is_primitive() and arg.is_function() and arg.res().is_function()):
            return
        arg = innermostFunction(arg)
        subs = function.can_unify(arg.arg())
        if subs is not None:
            xcat = arg.res().substitute(subs)
            yield FunctionalCategory(xcat, FunctionalCategory(xcat, function, arg.dir()), -arg.dir())

    def __str__(self):
        return 'T'