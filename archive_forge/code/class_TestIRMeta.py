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
class TestIRMeta(CheckEquality):
    """
    Tests IR node meta, like Loc and Scope
    """

    def test_loc(self):
        a = ir.Loc('file', 1, 0)
        b = ir.Loc('file', 1, 0)
        c = ir.Loc('pile', 1, 0)
        d = ir.Loc('file', 2, 0)
        e = ir.Loc('file', 1, 1)
        self.check(a, same=[b], different=[c, d, e])
        f = ir.Loc('file', 1, 0, maybe_decorator=False)
        g = ir.Loc('file', 1, 0, maybe_decorator=True)
        self.check(a, same=[f, g])

    def test_scope(self):
        parent1 = ir.Scope(None, self.loc1)
        parent2 = ir.Scope(None, self.loc1)
        parent3 = ir.Scope(None, self.loc2)
        self.check(parent1, same=[parent2, parent3])
        a = ir.Scope(parent1, self.loc1)
        b = ir.Scope(parent1, self.loc1)
        c = ir.Scope(parent1, self.loc2)
        d = ir.Scope(parent3, self.loc1)
        self.check(a, same=[b, c, d])
        e = ir.Scope(parent2, self.loc1)
        self.check(a, same=[e])