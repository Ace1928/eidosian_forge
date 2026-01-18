from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
class TestCustomPipeline(TestCase):

    def setUp(self):
        super(TestCustomPipeline, self).setUp()

        class CustomPipeline(Compiler):
            custom_pipeline_cache = []

            def compile_extra(self, func):
                self.custom_pipeline_cache.append(func)
                return super(CustomPipeline, self).compile_extra(func)

            def compile_ir(self, func_ir, *args, **kwargs):
                self.custom_pipeline_cache.append(func_ir)
                return super(CustomPipeline, self).compile_ir(func_ir, *args, **kwargs)
        self.pipeline_class = CustomPipeline

    def test_jit_custom_pipeline(self):
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

        @jit(pipeline_class=self.pipeline_class)
        def foo(x):
            return x
        self.assertEqual(foo(4), 4)
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [foo.py_func])

    def test_cfunc_custom_pipeline(self):
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

        @cfunc(types.int64(types.int64), pipeline_class=self.pipeline_class)
        def foo(x):
            return x
        self.assertEqual(foo(4), 4)
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [foo.__wrapped__])

    def test_objmode_custom_pipeline(self):
        self.assertListEqual(self.pipeline_class.custom_pipeline_cache, [])

        @jit(pipeline_class=self.pipeline_class)
        def foo(x):
            with objmode(x='intp'):
                x += int(1)
            return x
        arg = 123
        self.assertEqual(foo(arg), arg + 1)
        self.assertEqual(len(self.pipeline_class.custom_pipeline_cache), 2)
        first = self.pipeline_class.custom_pipeline_cache[0]
        self.assertIs(first, foo.py_func)
        second = self.pipeline_class.custom_pipeline_cache[1]
        self.assertIsInstance(second, FunctionIR)