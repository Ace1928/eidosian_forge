from __future__ import annotations
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin, N
from sympy.core.numbers import oo
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.matrices import eye
from sympy.multipledispatch import dispatch
from sympy.printing import sstr
from sympy.sets import Set, Union, FiniteSet
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets.handlers.union import union_sets
from sympy.solvers.solvers import solve
from sympy.utilities.misc import func_name
from sympy.utilities.iterables import is_sequence
def _svg(self, scale_factor=1.0, fill_color='#66cc99'):
    """Returns SVG path element for the GeometryEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
    raise NotImplementedError()