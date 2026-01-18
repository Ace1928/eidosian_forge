import numpy as np
from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, \
def _sweep_poly_phase(t, poly):
    """
    Calculate the phase used by sweep_poly to generate its output.

    See `sweep_poly` for a description of the arguments.

    """
    intpoly = polyint(poly)
    phase = 2 * pi * polyval(intpoly, t)
    return phase