from sympy.core import cacheit, Dummy, Ne, Integer, Rational, S, Wild
from sympy.functions import binomial, sin, cos, Piecewise, Abs
from .integrals import integrate
@cacheit
def _pat_sincos(x):
    a = Wild('a', exclude=[x])
    n, m = [Wild(s, exclude=[x], properties=[_integer_instance]) for s in 'nm']
    pat = sin(a * x) ** n * cos(a * x) ** m
    return (pat, a, n, m)