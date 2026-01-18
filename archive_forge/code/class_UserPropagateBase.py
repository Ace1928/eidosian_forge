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
class UserPropagateBase:

    def __init__(self, s, ctx=None):
        assert s is None or ctx is None
        ensure_prop_closures()
        self.solver = s
        self._ctx = None
        self.fresh_ctx = None
        self.cb = None
        self.id = _prop_closures.insert(self)
        self.fixed = None
        self.final = None
        self.eq = None
        self.diseq = None
        self.created = None
        if ctx:
            self.fresh_ctx = ctx
        if s:
            Z3_solver_propagate_init(self.ctx_ref(), s.solver, ctypes.c_void_p(self.id), _user_prop_push, _user_prop_pop, _user_prop_fresh)

    def __del__(self):
        if self._ctx:
            self._ctx.ctx = None

    def ctx(self):
        if self.fresh_ctx:
            return self.fresh_ctx
        else:
            return self.solver.ctx

    def ctx_ref(self):
        return self.ctx().ref()

    def add_fixed(self, fixed):
        assert not self.fixed
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_fixed(self.ctx_ref(), self.solver.solver, _user_prop_fixed)
        self.fixed = fixed

    def add_created(self, created):
        assert not self.created
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_created(self.ctx_ref(), self.solver.solver, _user_prop_created)
        self.created = created

    def add_final(self, final):
        assert not self.final
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_final(self.ctx_ref(), self.solver.solver, _user_prop_final)
        self.final = final

    def add_eq(self, eq):
        assert not self.eq
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_eq(self.ctx_ref(), self.solver.solver, _user_prop_eq)
        self.eq = eq

    def add_diseq(self, diseq):
        assert not self.diseq
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_diseq(self.ctx_ref(), self.solver.solver, _user_prop_diseq)
        self.diseq = diseq

    def add_decide(self, decide):
        assert not self.decide
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_decide(self.ctx_ref(), self.solver.solver, _user_prop_decide)
        self.decide = decide

    def push(self):
        raise Z3Exception('push needs to be overwritten')

    def pop(self, num_scopes):
        raise Z3Exception('pop needs to be overwritten')

    def fresh(self, new_ctx):
        raise Z3Exception('fresh needs to be overwritten')

    def add(self, e):
        assert not self._ctx
        if self.solver:
            Z3_solver_propagate_register(self.ctx_ref(), self.solver.solver, e.ast)
        else:
            Z3_solver_propagate_register_cb(self.ctx_ref(), ctypes.c_void_p(self.cb), e.ast)

    def next_split(self, t, idx, phase):
        return Z3_solver_next_split(self.ctx_ref(), ctypes.c_void_p(self.cb), t.ast, idx, phase)

    def propagate(self, e, ids, eqs=[]):
        _ids, num_fixed = _to_ast_array(ids)
        num_eqs = len(eqs)
        _lhs, _num_lhs = _to_ast_array([x for x, y in eqs])
        _rhs, _num_rhs = _to_ast_array([y for x, y in eqs])
        return Z3_solver_propagate_consequence(e.ctx.ref(), ctypes.c_void_p(self.cb), num_fixed, _ids, num_eqs, _lhs, _rhs, e.ast)

    def conflict(self, deps=[], eqs=[]):
        self.propagate(BoolVal(False, self.ctx()), deps, eqs)