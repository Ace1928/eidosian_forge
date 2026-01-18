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
class TestIRCompounds(CheckEquality):
    """
    Tests IR concepts that have state
    """

    def test_varmap(self):
        a = ir.VarMap()
        a.define(self.var_a, 'foo')
        a.define(self.var_b, 'bar')
        b = ir.VarMap()
        b.define(self.var_a, 'foo')
        b.define(self.var_b, 'bar')
        c = ir.VarMap()
        c.define(self.var_a, 'foo')
        c.define(self.var_c, 'bar')
        self.check(a, same=[b], different=[c])

    def test_block(self):

        def gen_block():
            parent = ir.Scope(None, self.loc1)
            tmp = ir.Block(parent, self.loc2)
            assign1 = ir.Assign(self.var_a, self.var_b, self.loc3)
            assign2 = ir.Assign(self.var_a, self.var_c, self.loc3)
            assign3 = ir.Assign(self.var_c, self.var_b, self.loc3)
            tmp.append(assign1)
            tmp.append(assign2)
            tmp.append(assign3)
            return tmp
        a = gen_block()
        b = gen_block()
        c = gen_block().append(ir.Assign(self.var_a, self.var_b, self.loc3))
        self.check(a, same=[b], different=[c])

    def test_functionir(self):

        def run_frontend(x):
            return compiler.run_frontend(x, emit_dels=True)

        def gen():
            _FREEVAR = 51966

            def foo(a, b, c=12, d=1j, e=None):
                f = a + b
                a += _FREEVAR
                g = np.zeros(c, dtype=np.complex64)
                h = f + g
                i = 1j / d
                if np.abs(i) > 0:
                    k = h / i
                    l = np.arange(1, c + 1)
                    with objmode():
                        print(e, k)
                    m = np.sqrt(l - g)
                    if np.abs(m[0]) < 1:
                        n = 0
                        for o in range(a):
                            n += 0
                            if np.abs(n) < 3:
                                break
                        n += m[2]
                    p = g / l
                    q = []
                    for r in range(len(p)):
                        q.append(p[r])
                        if r > 4 + 1:
                            with objmode(s='intp', t='complex128'):
                                s = 123
                                t = 5
                            if s > 122:
                                t += s
                        t += q[0] + _GLOBAL
                return f + o + r + t + r + a + n
            return foo
        x = gen()
        y = gen()
        x_ir = run_frontend(x)
        y_ir = run_frontend(y)
        self.assertTrue(x_ir.equal_ir(y_ir))

        def check_diffstr(string, pointing_at=[]):
            lines = string.splitlines()
            for item in pointing_at:
                for l in lines:
                    if l.startswith('->'):
                        if item in l:
                            break
                else:
                    raise AssertionError('Could not find %s ' % item)
        self.assertIn('IR is considered equivalent', x_ir.diff_str(y_ir))
        for label in reversed(list(y_ir.blocks.keys())):
            blk = y_ir.blocks[label]
            if isinstance(blk.body[-1], ir.Branch):
                ref = blk.body[-1]
                ref.truebr, ref.falsebr = (ref.falsebr, ref.truebr)
                break
        check_diffstr(x_ir.diff_str(y_ir), ['branch'])
        z = gen()
        self.assertFalse(x_ir.equal_ir(y_ir))
        z_ir = run_frontend(z)
        change_set = set()
        for label in reversed(list(z_ir.blocks.keys())):
            blk = z_ir.blocks[label]
            ref = blk.body[:-1]
            idx = None
            for i in range(len(ref) - 1):
                if isinstance(ref[i], ir.Del) and isinstance(ref[i + 1], ir.Del):
                    idx = i
                    break
            if idx is not None:
                b = blk.body
                change_set.add(str(b[idx + 1]))
                change_set.add(str(b[idx]))
                b[idx], b[idx + 1] = (b[idx + 1], b[idx])
                break
        self.assertTrue(change_set)
        self.assertFalse(x_ir.equal_ir(z_ir))
        self.assertEqual(len(change_set), 2)
        for item in change_set:
            self.assertTrue(item.startswith('del '))
        check_diffstr(x_ir.diff_str(z_ir), change_set)

        def foo(a, b):
            c = a * 2
            d = c + b
            e = np.sqrt(d)
            return e

        def bar(a, b):
            c = a * 2
            d = c + b
            e = np.sqrt(d)
            return e

        def baz(a, b):
            c = a * 2
            d = b + c
            e = np.sqrt(d + 1)
            return e
        foo_ir = run_frontend(foo)
        bar_ir = run_frontend(bar)
        self.assertTrue(foo_ir.equal_ir(bar_ir))
        self.assertIn('IR is considered equivalent', foo_ir.diff_str(bar_ir))
        baz_ir = run_frontend(baz)
        self.assertFalse(foo_ir.equal_ir(baz_ir))
        tmp = foo_ir.diff_str(baz_ir)
        self.assertIn('Other block contains more statements', tmp)
        check_diffstr(tmp, ['c + b', 'b + c'])