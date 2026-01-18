from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class Tactic:
    """Tactics transform, solver and/or simplify sets of constraints (Goal).
    A Tactic can be converted into a Solver using the method solver().

    Several combinators are available for creating new tactics using the built-in ones:
    Then(), OrElse(), FailIf(), Repeat(), When(), Cond().
    """

    def __init__(self, tactic, ctx=None):
        self.ctx = _get_ctx(ctx)
        self.tactic = None
        if isinstance(tactic, TacticObj):
            self.tactic = tactic
        else:
            if z3_debug():
                _z3_assert(isinstance(tactic, str), 'tactic name expected')
            try:
                self.tactic = Z3_mk_tactic(self.ctx.ref(), str(tactic))
            except Z3Exception:
                raise Z3Exception("unknown tactic '%s'" % tactic)
        Z3_tactic_inc_ref(self.ctx.ref(), self.tactic)

    def __deepcopy__(self, memo={}):
        return Tactic(self.tactic, self.ctx)

    def __del__(self):
        if self.tactic is not None and self.ctx.ref() is not None and (Z3_tactic_dec_ref is not None):
            Z3_tactic_dec_ref(self.ctx.ref(), self.tactic)

    def solver(self, logFile=None):
        """Create a solver using the tactic `self`.

        The solver supports the methods `push()` and `pop()`, but it
        will always solve each `check()` from scratch.

        >>> t = Then('simplify', 'nlsat')
        >>> s = t.solver()
        >>> x = Real('x')
        >>> s.add(x**2 == 2, x > 0)
        >>> s.check()
        sat
        >>> s.model()
        [x = 1.4142135623?]
        """
        return Solver(Z3_mk_solver_from_tactic(self.ctx.ref(), self.tactic), self.ctx, logFile)

    def apply(self, goal, *arguments, **keywords):
        """Apply tactic `self` to the given goal or Z3 Boolean expression using the given options.

        >>> x, y = Ints('x y')
        >>> t = Tactic('solve-eqs')
        >>> t.apply(And(x == 0, y >= x + 1))
        [[y >= 1]]
        """
        if z3_debug():
            _z3_assert(isinstance(goal, (Goal, BoolRef)), 'Z3 Goal or Boolean expressions expected')
        goal = _to_goal(goal)
        if len(arguments) > 0 or len(keywords) > 0:
            p = args2params(arguments, keywords, self.ctx)
            return ApplyResult(Z3_tactic_apply_ex(self.ctx.ref(), self.tactic, goal.goal, p.params), self.ctx)
        else:
            return ApplyResult(Z3_tactic_apply(self.ctx.ref(), self.tactic, goal.goal), self.ctx)

    def __call__(self, goal, *arguments, **keywords):
        """Apply tactic `self` to the given goal or Z3 Boolean expression using the given options.

        >>> x, y = Ints('x y')
        >>> t = Tactic('solve-eqs')
        >>> t(And(x == 0, y >= x + 1))
        [[y >= 1]]
        """
        return self.apply(goal, *arguments, **keywords)

    def help(self):
        """Display a string containing a description of the available options for the `self` tactic."""
        print(Z3_tactic_get_help(self.ctx.ref(), self.tactic))

    def param_descrs(self):
        """Return the parameter description set."""
        return ParamDescrsRef(Z3_tactic_get_param_descrs(self.ctx.ref(), self.tactic), self.ctx)