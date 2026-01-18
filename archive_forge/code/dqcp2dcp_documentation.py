from collections import namedtuple
from typing import Any, List, Tuple
import numpy as np
from cvxpy import problems
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.max import max as max_atom
from cvxpy.atoms.min import min as min_atom
from cvxpy.constraints import Inequality
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
from cvxpy.reductions.dqcp2dcp import inverse, sets, tighten
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution
Recursively canonicalize a constraint.

        The DQCP grammar has expresions of the form

            INCR* QCVX DCP

        and

            DECR* QCCV DCP

        ie, zero or more real/scalar increasing (or decreasing) atoms, composed
        with a quasiconvex (or quasiconcave) atom, composed with DCP
        expressions.

        The monotone functions are inverted by applying their inverses to
        both sides of a constraint. The QCVX (QCCV) atom is lowered by
        replacing it with its sublevel (superlevel) set. The DCP
        expressions are canonicalized via graph implementations.
        