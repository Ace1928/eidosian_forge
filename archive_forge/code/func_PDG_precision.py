from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def PDG_precision(std_dev):
    """
    Return the number of significant digits to be used for the given
    standard deviation, according to the rounding rules of the
    Particle Data Group (2010)
    (http://pdg.lbl.gov/2010/reviews/rpp2010-rev-rpp-intro.pdf).

    Also returns the effective standard deviation to be used for
    display.
    """
    exponent = first_digit(std_dev)
    if exponent >= 0:
        exponent, factor = (exponent - 2, 1)
    else:
        exponent, factor = (exponent + 1, 1000)
    digits = int(std_dev / 10.0 ** exponent * factor)
    if digits <= 354:
        return (2, std_dev)
    elif digits <= 949:
        return (1, std_dev)
    else:
        return (2, 10.0 ** exponent * (1000 / factor))