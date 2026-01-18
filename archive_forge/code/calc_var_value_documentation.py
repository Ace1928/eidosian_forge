from pyomo.common.errors import IterationLimitError
from pyomo.common.numeric_types import native_numeric_types, native_complex_types, value
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.base.constraint import Constraint, _ConstraintData
import logging
Calculate the variable value given a specified equality constraint

    This function calculates the value of the specified variable
    necessary to make the provided equality constraint feasible
    (assuming any other variables values are fixed).  The method first
    attempts to solve for the variable value assuming it appears
    linearly in the constraint.  If that doesn't converge the constraint
    residual, it falls back on Newton's method using exact (symbolic)
    derivatives.

    Notes
    -----
    This is an unconstrained solver and is NOT guaranteed to respect the
    variable bounds or domain.  The solver may leave the variable value
    in an infeasible state (outside the declared bounds or domain bounds).

    Parameters:
    -----------
    variable: :py:class:`_VarData`
        The variable to solve for
    constraint: :py:class:`_ConstraintData` or relational expression or `tuple`
        The equality constraint to use to solve for the variable value.
        May be a `ConstraintData` object or any valid argument for
        ``Constraint(expr=<>)`` (i.e., a relational expression or 2- or
        3-tuple)
    eps: `float`
        The tolerance to use to determine equality [default=1e-8].
    iterlim: `int`
        The maximum number of iterations if this method has to fall back
        on using Newton's method.  Raises RuntimeError on iteration
        limit [default=1000]
    linesearch: `bool`
        Decides whether or not to use the linesearch (recommended).
        [default=True]
    alpha_min: `float`
        The minimum fractional step to use in the linesearch [default=1e-8].
    diff_mode: :py:enum:`pyomo.core.expr.calculus.derivatives.Modes`
        The mode to use to differentiate the expression.  If
        unspecified, defaults to `Modes.sympy`

    Returns:
    --------
    None

    