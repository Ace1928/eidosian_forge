import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial

    Use CyPHC to find the shapes corresponding to SL2C representations
    of the given closed manifold, as well as those which are
    boundary-parabolic with respect to the Dehn-filling description.

    sage: M = Manifold('m006(-5, 1)')
    sage: shape_sets = shapes_of_SL2C_reps_for_filled(M)
    sage: len(shape_sets)
    24
    sage: max(shapes['err'] for shapes in shape_sets) < 1e-13
    True
    