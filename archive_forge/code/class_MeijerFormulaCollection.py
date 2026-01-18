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
class MeijerFormulaCollection:
    """
    This class holds a collection of meijer g formulae.
    """

    def __init__(self):
        formulae = []
        add_meijerg_formulae(formulae)
        self.formulae = defaultdict(list)
        for formula in formulae:
            self.formulae[formula.func.signature].append(formula)
        self.formulae = dict(self.formulae)

    def lookup_origin(self, func):
        """ Try to find a formula that matches func. """
        if func.signature not in self.formulae:
            return None
        for formula in self.formulae[func.signature]:
            res = formula.try_instantiate(func)
            if res is not None:
                return res