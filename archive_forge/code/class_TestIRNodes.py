import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
class TestIRNodes(CheckEquality):
    """
    Tests IR nodes
    """

    def test_terminator(self):
        t1 = ir.Terminator()
        t2 = ir.Terminator()
        self.check(t1, same=[t2])

    def test_jump(self):
        a = ir.Jump(1, self.loc1)
        b = ir.Jump(1, self.loc1)
        c = ir.Jump(1, self.loc2)
        d = ir.Jump(2, self.loc1)
        self.check(a, same=[b, c], different=[d])

    def test_return(self):
        a = ir.Return(self.var_a, self.loc1)
        b = ir.Return(self.var_a, self.loc1)
        c = ir.Return(self.var_a, self.loc2)
        d = ir.Return(self.var_b, self.loc1)
        self.check(a, same=[b, c], different=[d])

    def test_raise(self):
        a = ir.Raise(self.var_a, self.loc1)
        b = ir.Raise(self.var_a, self.loc1)
        c = ir.Raise(self.var_a, self.loc2)
        d = ir.Raise(self.var_b, self.loc1)
        self.check(a, same=[b, c], different=[d])

    def test_staticraise(self):
        a = ir.StaticRaise(AssertionError, None, self.loc1)
        b = ir.StaticRaise(AssertionError, None, self.loc1)
        c = ir.StaticRaise(AssertionError, None, self.loc2)
        e = ir.StaticRaise(AssertionError, ('str',), self.loc1)
        d = ir.StaticRaise(RuntimeError, None, self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_branch(self):
        a = ir.Branch(self.var_a, 1, 2, self.loc1)
        b = ir.Branch(self.var_a, 1, 2, self.loc1)
        c = ir.Branch(self.var_a, 1, 2, self.loc2)
        d = ir.Branch(self.var_b, 1, 2, self.loc1)
        e = ir.Branch(self.var_a, 2, 2, self.loc1)
        f = ir.Branch(self.var_a, 1, 3, self.loc1)
        self.check(a, same=[b, c], different=[d, e, f])

    def test_expr(self):
        a = ir.Expr('some_op', self.loc1)
        b = ir.Expr('some_op', self.loc1)
        c = ir.Expr('some_op', self.loc2)
        d = ir.Expr('some_other_op', self.loc1)
        self.check(a, same=[b, c], different=[d])

    def test_setitem(self):
        a = ir.SetItem(self.var_a, self.var_b, self.var_c, self.loc1)
        b = ir.SetItem(self.var_a, self.var_b, self.var_c, self.loc1)
        c = ir.SetItem(self.var_a, self.var_b, self.var_c, self.loc2)
        d = ir.SetItem(self.var_d, self.var_b, self.var_c, self.loc1)
        e = ir.SetItem(self.var_a, self.var_d, self.var_c, self.loc1)
        f = ir.SetItem(self.var_a, self.var_b, self.var_d, self.loc1)
        self.check(a, same=[b, c], different=[d, e, f])

    def test_staticsetitem(self):
        a = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_c, self.loc1)
        b = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_c, self.loc1)
        c = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_c, self.loc2)
        d = ir.StaticSetItem(self.var_d, 1, self.var_b, self.var_c, self.loc1)
        e = ir.StaticSetItem(self.var_a, 2, self.var_b, self.var_c, self.loc1)
        f = ir.StaticSetItem(self.var_a, 1, self.var_d, self.var_c, self.loc1)
        g = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_d, self.loc1)
        self.check(a, same=[b, c], different=[d, e, f, g])

    def test_delitem(self):
        a = ir.DelItem(self.var_a, self.var_b, self.loc1)
        b = ir.DelItem(self.var_a, self.var_b, self.loc1)
        c = ir.DelItem(self.var_a, self.var_b, self.loc2)
        d = ir.DelItem(self.var_c, self.var_b, self.loc1)
        e = ir.DelItem(self.var_a, self.var_c, self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_del(self):
        a = ir.Del(self.var_a.name, self.loc1)
        b = ir.Del(self.var_a.name, self.loc1)
        c = ir.Del(self.var_a.name, self.loc2)
        d = ir.Del(self.var_b.name, self.loc1)
        self.check(a, same=[b, c], different=[d])

    def test_setattr(self):
        a = ir.SetAttr(self.var_a, 'foo', self.var_b, self.loc1)
        b = ir.SetAttr(self.var_a, 'foo', self.var_b, self.loc1)
        c = ir.SetAttr(self.var_a, 'foo', self.var_b, self.loc2)
        d = ir.SetAttr(self.var_c, 'foo', self.var_b, self.loc1)
        e = ir.SetAttr(self.var_a, 'bar', self.var_b, self.loc1)
        f = ir.SetAttr(self.var_a, 'foo', self.var_c, self.loc1)
        self.check(a, same=[b, c], different=[d, e, f])

    def test_delattr(self):
        a = ir.DelAttr(self.var_a, 'foo', self.loc1)
        b = ir.DelAttr(self.var_a, 'foo', self.loc1)
        c = ir.DelAttr(self.var_a, 'foo', self.loc2)
        d = ir.DelAttr(self.var_c, 'foo', self.loc1)
        e = ir.DelAttr(self.var_a, 'bar', self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_assign(self):
        a = ir.Assign(self.var_a, self.var_b, self.loc1)
        b = ir.Assign(self.var_a, self.var_b, self.loc1)
        c = ir.Assign(self.var_a, self.var_b, self.loc2)
        d = ir.Assign(self.var_c, self.var_b, self.loc1)
        e = ir.Assign(self.var_a, self.var_c, self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_print(self):
        a = ir.Print((self.var_a,), self.var_b, self.loc1)
        b = ir.Print((self.var_a,), self.var_b, self.loc1)
        c = ir.Print((self.var_a,), self.var_b, self.loc2)
        d = ir.Print((self.var_c,), self.var_b, self.loc1)
        e = ir.Print((self.var_a,), self.var_c, self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_storemap(self):
        a = ir.StoreMap(self.var_a, self.var_b, self.var_c, self.loc1)
        b = ir.StoreMap(self.var_a, self.var_b, self.var_c, self.loc1)
        c = ir.StoreMap(self.var_a, self.var_b, self.var_c, self.loc2)
        d = ir.StoreMap(self.var_d, self.var_b, self.var_c, self.loc1)
        e = ir.StoreMap(self.var_a, self.var_d, self.var_c, self.loc1)
        f = ir.StoreMap(self.var_a, self.var_b, self.var_d, self.loc1)
        self.check(a, same=[b, c], different=[d, e, f])

    def test_yield(self):
        a = ir.Yield(self.var_a, self.loc1, 0)
        b = ir.Yield(self.var_a, self.loc1, 0)
        c = ir.Yield(self.var_a, self.loc2, 0)
        d = ir.Yield(self.var_b, self.loc1, 0)
        e = ir.Yield(self.var_a, self.loc1, 1)
        self.check(a, same=[b, c], different=[d, e])

    def test_enterwith(self):
        a = ir.EnterWith(self.var_a, 0, 1, self.loc1)
        b = ir.EnterWith(self.var_a, 0, 1, self.loc1)
        c = ir.EnterWith(self.var_a, 0, 1, self.loc2)
        d = ir.EnterWith(self.var_b, 0, 1, self.loc1)
        e = ir.EnterWith(self.var_a, 1, 1, self.loc1)
        f = ir.EnterWith(self.var_a, 0, 2, self.loc1)
        self.check(a, same=[b, c], different=[d, e, f])

    def test_arg(self):
        a = ir.Arg('foo', 0, self.loc1)
        b = ir.Arg('foo', 0, self.loc1)
        c = ir.Arg('foo', 0, self.loc2)
        d = ir.Arg('bar', 0, self.loc1)
        e = ir.Arg('foo', 1, self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_const(self):
        a = ir.Const(1, self.loc1)
        b = ir.Const(1, self.loc1)
        c = ir.Const(1, self.loc2)
        d = ir.Const(2, self.loc1)
        self.check(a, same=[b, c], different=[d])

    def test_global(self):
        a = ir.Global('foo', 0, self.loc1)
        b = ir.Global('foo', 0, self.loc1)
        c = ir.Global('foo', 0, self.loc2)
        d = ir.Global('bar', 0, self.loc1)
        e = ir.Global('foo', 1, self.loc1)
        self.check(a, same=[b, c], different=[d, e])

    def test_var(self):
        a = ir.Var(None, 'foo', self.loc1)
        b = ir.Var(None, 'foo', self.loc1)
        c = ir.Var(None, 'foo', self.loc2)
        d = ir.Var(ir.Scope(None, ir.unknown_loc), 'foo', self.loc1)
        e = ir.Var(None, 'bar', self.loc1)
        self.check(a, same=[b, c, d], different=[e])

    def test_undefinedtype(self):
        a = ir.UndefinedType()
        b = ir.UndefinedType()
        self.check(a, same=[b])

    def test_loop(self):
        a = ir.Loop(1, 3)
        b = ir.Loop(1, 3)
        c = ir.Loop(2, 3)
        d = ir.Loop(1, 4)
        self.check(a, same=[b], different=[c, d])

    def test_with(self):
        a = ir.With(1, 3)
        b = ir.With(1, 3)
        c = ir.With(2, 3)
        d = ir.With(1, 4)
        self.check(a, same=[b], different=[c, d])