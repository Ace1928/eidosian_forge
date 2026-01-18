from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
from sympy.core.mod import Mod
from sympy.core.numbers import igcd, I, Number, Rational, oo, ilcm
from sympy.core.power import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType
def _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset, original_imageset, newresult, eq=None):
    """If `rnew` (A dict <symbol: soln>) contains valid soln
        append it to `newresult` list.
        `imgset_yes` is (base, dummy_var) if there was imageset in previously
         calculated result(otherwise empty tuple). `original_imageset` is dict
         of imageset expr and imageset from this result.
        `soln_imageset` dict of imageset expr and imageset of new soln.
        """
    satisfy_exclude = _check_exclude(rnew, imgset_yes)
    delete_soln = False
    if not satisfy_exclude:
        local_n = None
        if imgset_yes:
            local_n = imgset_yes[0]
            base = imgset_yes[1]
            if sym and sol:
                dummy_list = list(sol.atoms(Dummy))
                local_n_list = [local_n for i in range(0, len(dummy_list))]
                dummy_zip = zip(dummy_list, local_n_list)
                lam = Lambda(local_n, sol.subs(dummy_zip))
                rnew[sym] = ImageSet(lam, base)
            if eq is not None:
                newresult, rnew, delete_soln = _append_eq(eq, newresult, rnew, delete_soln, local_n)
        elif eq is not None:
            newresult, rnew, delete_soln = _append_eq(eq, newresult, rnew, delete_soln)
        elif sol in soln_imageset.keys():
            rnew[sym] = soln_imageset[sol]
            _restore_imgset(rnew, original_imageset, newresult)
        else:
            newresult.append(rnew)
    elif satisfy_exclude:
        delete_soln = True
        rnew = {}
    _restore_imgset(rnew, original_imageset, newresult)
    return (newresult, delete_soln)