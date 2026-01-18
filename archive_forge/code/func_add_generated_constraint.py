import dataclasses
import datetime
import enum
import math
from typing import Dict, List, Mapping, Optional, Set, Union
from ortools.math_opt import callback_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def add_generated_constraint(self, bounded_expr: Optional[Union[bool, model.BoundedLinearTypes]]=None, *, lb: Optional[float]=None, ub: Optional[float]=None, expr: Optional[model.LinearTypes]=None, is_lazy: bool) -> None:
    """Adds a linear constraint to the list of generated constraints.

        The constraint can be of two exclusive types: a "lazy constraint" or a
        "user cut. A "user cut" is a constraint that excludes the current LP
        solution, but does not cut off any integer-feasible points that satisfy the
        already added constraints (either in callbacks or through
        Model.add_linear_constraint()). A "lazy constraint" is a constraint that
        excludes such integer-feasible points and hence is needed for corrctness of
        the forlumation.

        The simplest way to specify the constraint is by passing a one-sided or
        two-sided linear inequality as in:
          * add_generated_constraint(x + y + 1.0 <= 2.0, is_lazy=True),
          * add_generated_constraint(x + y >= 2.0, is_lazy=True), or
          * add_generated_constraint((1.0 <= x + y) <= 2.0, is_lazy=True).

        Note the extra parenthesis for two-sided linear inequalities, which is
        required due to some language limitations (see
        https://peps.python.org/pep-0335/ and https://peps.python.org/pep-0535/).
        If the parenthesis are omitted, a TypeError will be raised explaining the
        issue (if this error was not raised the first inequality would have been
        silently ignored because of the noted language limitations).

        The second way to specify the constraint is by setting lb, ub, and/o expr as
        in:
          * add_generated_constraint(expr=x + y + 1.0, ub=2.0, is_lazy=True),
          * add_generated_constraint(expr=x + y, lb=2.0, is_lazy=True),
          * add_generated_constraint(expr=x + y, lb=1.0, ub=2.0, is_lazy=True), or
          * add_generated_constraint(lb=1.0, is_lazy=True).
        Omitting lb is equivalent to setting it to -math.inf and omiting ub is
        equivalent to setting it to math.inf.

        These two alternatives are exclusive and a combined call like:
          * add_generated_constraint(x + y <= 2.0, lb=1.0, is_lazy=True), or
          * add_generated_constraint(x + y <= 2.0, ub=math.inf, is_lazy=True)
        will raise a ValueError. A ValueError is also raised if expr's offset is
        infinite.

        Args:
          bounded_expr: a linear inequality describing the constraint. Cannot be
            specified together with lb, ub, or expr.
          lb: The constraint's lower bound if bounded_expr is omitted (if both
            bounder_expr and lb are omitted, the lower bound is -math.inf).
          ub: The constraint's upper bound if bounded_expr is omitted (if both
            bounder_expr and ub are omitted, the upper bound is math.inf).
          expr: The constraint's linear expression if bounded_expr is omitted.
          is_lazy: Whether the constraint is lazy or not.
        """
    normalized_inequality = model.as_normalized_linear_inequality(bounded_expr, lb=lb, ub=ub, expr=expr)
    self.generated_constraints.append(GeneratedConstraint(lower_bound=normalized_inequality.lb, terms=normalized_inequality.coefficients, upper_bound=normalized_inequality.ub, is_lazy=is_lazy))