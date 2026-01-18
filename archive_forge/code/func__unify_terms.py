import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _unify_terms(a, b, bindings=None, used=None):
    """
    This method attempts to unify two terms.  Two expressions are unifiable
    if there exists a substitution function S such that S(a) == S(-b).

    :param a: ``Expression``
    :param b: ``Expression``
    :param bindings: ``BindingDict`` a starting set of bindings with which
    the unification must be consistent
    :return: ``BindingDict`` A dictionary of the bindings required to unify
    :raise ``BindingException``: If the terms cannot be unified
    """
    assert isinstance(a, Expression)
    assert isinstance(b, Expression)
    if bindings is None:
        bindings = BindingDict()
    if used is None:
        used = ([], [])
    if isinstance(a, NegatedExpression) and isinstance(b, ApplicationExpression):
        newbindings = most_general_unification(a.term, b, bindings)
        newused = (used[0] + [a], used[1] + [b])
        unused = ([], [])
    elif isinstance(a, ApplicationExpression) and isinstance(b, NegatedExpression):
        newbindings = most_general_unification(a, b.term, bindings)
        newused = (used[0] + [a], used[1] + [b])
        unused = ([], [])
    elif isinstance(a, EqualityExpression):
        newbindings = BindingDict([(a.first.variable, a.second)])
        newused = (used[0] + [a], used[1])
        unused = ([], [b])
    elif isinstance(b, EqualityExpression):
        newbindings = BindingDict([(b.first.variable, b.second)])
        newused = (used[0], used[1] + [b])
        unused = ([a], [])
    else:
        raise BindingException((a, b))
    return (newbindings, newused, unused)