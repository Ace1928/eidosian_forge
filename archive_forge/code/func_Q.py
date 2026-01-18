from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def Q(bidx):
    return (set(), [bidx])