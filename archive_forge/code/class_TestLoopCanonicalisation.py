from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
class TestLoopCanonicalisation(MemoryLeakMixin, TestCase):

    def get_pipeline(use_canonicaliser, use_partial_typing=False):

        class NewCompiler(CompilerBase):

            def define_pipelines(self):
                pm = PassManager('custom_pipeline')
                pm.add_pass(TranslateByteCode, 'analyzing bytecode')
                pm.add_pass(IRProcessing, 'processing IR')
                pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
                if use_partial_typing:
                    pm.add_pass(PartialTypeInference, 'do partial typing')
                if use_canonicaliser:
                    pm.add_pass(IterLoopCanonicalization, 'Canonicalise loops')
                pm.add_pass(SimplifyCFG, 'Simplify the CFG')
                if use_partial_typing:
                    pm.add_pass(ResetTypeInfo, 'resets the type info state')
                pm.add_pass(NopythonTypeInference, 'nopython frontend')
                pm.add_pass(IRLegalization, 'ensure IR is legal')
                pm.add_pass(PreserveIR, 'save IR for later inspection')
                pm.add_pass(NativeLowering, 'native lowering')
                pm.add_pass(NoPythonBackend, 'nopython mode backend')
                pm.finalize()
                return [pm]
        return NewCompiler
    LoopIgnoringCompiler = get_pipeline(False)
    LoopCanonicalisingCompiler = get_pipeline(True)
    TypedLoopCanonicalisingCompiler = get_pipeline(True, True)

    def test_simple_loop_in_depth(self):
        """ This heavily checks a simple loop transform """

        def get_info(pipeline):

            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for i in tup:
                    acc += i
                return acc
            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return (func_ir, cres.fndesc)
        ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.LoopCanonicalisingCompiler)

        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)
        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3, len(canonicalise_loops_fndesc.calltypes))

        def find_getX(fd, op):
            return [x for x in fd.calltypes.keys() if isinstance(x, ir.Expr) and x.op == op]
        il_getiters = find_getX(ignore_loops_fndesc, 'getiter')
        self.assertEqual(len(il_getiters), 1)
        cl_getiters = find_getX(canonicalise_loops_fndesc, 'getiter')
        self.assertEqual(len(cl_getiters), 1)
        cl_getitems = find_getX(canonicalise_loops_fndesc, 'getitem')
        self.assertEqual(len(cl_getitems), 1)
        self.assertEqual(il_getiters[0].value.name, cl_getitems[0].value.name)
        range_inst = canonicalise_loops_fndesc.calltypes[cl_getiters[0]].args[0]
        self.assertTrue(isinstance(range_inst, types.RangeType))

    def test_transform_scope(self):
        """ This checks the transform, when there's no typemap, will happily
        transform a loop on something that's not tuple-like
        """

        def get_info(pipeline):

            @njit(pipeline_class=pipeline)
            def foo():
                acc = 0
                for i in [1, 2, 3]:
                    acc += i
                return acc
            self.assertEqual(foo(), foo.py_func())
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return (func_ir, cres.fndesc)
        ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.LoopCanonicalisingCompiler)

        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)
        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3, len(canonicalise_loops_fndesc.calltypes))

        def find_getX(fd, op):
            return [x for x in fd.calltypes.keys() if isinstance(x, ir.Expr) and x.op == op]
        il_getiters = find_getX(ignore_loops_fndesc, 'getiter')
        self.assertEqual(len(il_getiters), 1)
        cl_getiters = find_getX(canonicalise_loops_fndesc, 'getiter')
        self.assertEqual(len(cl_getiters), 1)
        cl_getitems = find_getX(canonicalise_loops_fndesc, 'getitem')
        self.assertEqual(len(cl_getitems), 1)
        self.assertEqual(il_getiters[0].value.name, cl_getitems[0].value.name)
        range_inst = canonicalise_loops_fndesc.calltypes[cl_getiters[0]].args[0]
        self.assertTrue(isinstance(range_inst, types.RangeType))

    @unittest.skip('Waiting for pass to be enabled for all tuples')
    def test_influence_of_typed_transform(self):
        """ This heavily checks a typed transformation only impacts tuple
        induced loops"""

        def get_info(pipeline):

            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for i in range(4):
                    for y in tup:
                        for j in range(3):
                            acc += 1
                return acc
            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['func_ir']
            return (func_ir, cres.fndesc)
        ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.TypedLoopCanonicalisingCompiler)

        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)
        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3, len(canonicalise_loops_fndesc.calltypes))

        def find_getX(fd, op):
            return [x for x in fd.calltypes.keys() if isinstance(x, ir.Expr) and x.op == op]
        il_getiters = find_getX(ignore_loops_fndesc, 'getiter')
        self.assertEqual(len(il_getiters), 3)
        cl_getiters = find_getX(canonicalise_loops_fndesc, 'getiter')
        self.assertEqual(len(cl_getiters), 3)
        cl_getitems = find_getX(canonicalise_loops_fndesc, 'getitem')
        self.assertEqual(len(cl_getitems), 1)
        self.assertEqual(il_getiters[1].value.name, cl_getitems[0].value.name)
        for x in cl_getiters:
            range_inst = canonicalise_loops_fndesc.calltypes[x].args[0]
            self.assertTrue(isinstance(range_inst, types.RangeType))

    def test_influence_of_typed_transform_literal_unroll(self):
        """ This heavily checks a typed transformation only impacts loops with
        literal_unroll marker"""

        def get_info(pipeline):

            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for i in range(4):
                    for y in literal_unroll(tup):
                        for j in range(3):
                            acc += 1
                return acc
            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return (func_ir, cres.fndesc)
        ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.TypedLoopCanonicalisingCompiler)

        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)
        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3, len(canonicalise_loops_fndesc.calltypes))

        def find_getX(fd, op):
            return [x for x in fd.calltypes.keys() if isinstance(x, ir.Expr) and x.op == op]
        il_getiters = find_getX(ignore_loops_fndesc, 'getiter')
        self.assertEqual(len(il_getiters), 3)
        cl_getiters = find_getX(canonicalise_loops_fndesc, 'getiter')
        self.assertEqual(len(cl_getiters), 3)
        cl_getitems = find_getX(canonicalise_loops_fndesc, 'getitem')
        self.assertEqual(len(cl_getitems), 1)
        self.assertEqual(il_getiters[1].value.name, cl_getitems[0].value.name)
        for x in cl_getiters:
            range_inst = canonicalise_loops_fndesc.calltypes[x].args[0]
            self.assertTrue(isinstance(range_inst, types.RangeType))

    @unittest.skip('Waiting for pass to be enabled for all tuples')
    def test_lots_of_loops(self):
        """ This heavily checks a simple loop transform """

        def get_info(pipeline):

            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for i in tup:
                    acc += i
                    for j in tup + (4, 5, 6):
                        acc += 1 - j
                        if j > 5:
                            break
                    else:
                        acc -= 2
                for i in tup:
                    acc -= i % 2
                return acc
            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return (func_ir, cres.fndesc)
        ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.LoopCanonicalisingCompiler)

        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)
        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3 * 3, len(canonicalise_loops_fndesc.calltypes))

    def test_inlined_loops(self):
        """ Checks a loop appearing from a closure """

        def get_info(pipeline):

            @njit(pipeline_class=pipeline)
            def foo(tup):

                def bar(n):
                    acc = 0
                    for i in range(n):
                        acc += 1
                    return acc
                acc = 0
                for i in tup:
                    acc += i
                    acc += bar(i)
                return acc
            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return (func_ir, cres.fndesc)
        ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.LoopCanonicalisingCompiler)

        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)
        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 5, len(canonicalise_loops_fndesc.calltypes))