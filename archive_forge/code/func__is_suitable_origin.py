from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def _is_suitable_origin(self):
    """
        Decide if ``self`` is a suitable origin.

        Explanation
        ===========

        A function is a suitable origin iff:
        * none of the ai equals bj + n, with n a non-negative integer
        * none of the ai is zero
        * none of the bj is a non-positive integer

        Note that this gives meaningful results only when none of the indices
        are symbolic.

        """
    for a in self.ap:
        for b in self.bq:
            if (a - b).is_integer and (a - b).is_negative is False:
                return False
    for a in self.ap:
        if a == 0:
            return False
    for b in self.bq:
        if b.is_integer and b.is_nonpositive:
            return False
    return True