from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import (exp, log)
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.plot import LineOver1DRangeSeries
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex
def _check_system(system):
    """Function to check whether the dynamical system passed for plots is
    compatible or not."""
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError('Only SISO LTI systems are currently supported.')
    sys = system.to_expr()
    len_free_symbols = len(sys.free_symbols)
    if len_free_symbols > 1:
        raise ValueError('Extra degree of freedom found. Make sure that there are no free symbols in the dynamical system other than the variable of Laplace transform.')
    if sys.has(exp):
        raise NotImplementedError('Time delay terms are not supported.')