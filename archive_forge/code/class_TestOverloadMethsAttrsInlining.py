import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
class TestOverloadMethsAttrsInlining(InliningBase):

    def setUp(self):
        self.make_dummy_type()
        super(TestOverloadMethsAttrsInlining, self).setUp()

    def check_method(self, test_impl, args, expected, block_count, expects_inlined=True):
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(j_func(*args), expected)
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = fir.blocks
        self.assertEqual(len(fir.blocks), block_count)
        if expects_inlined:
            for block in fir.blocks.values():
                calls = list(block.find_exprs('call'))
                self.assertFalse(calls)
        else:
            allcalls = []
            for block in fir.blocks.values():
                allcalls += list(block.find_exprs('call'))
            self.assertTrue(allcalls)

    def check_getattr(self, test_impl, args, expected, block_count, expects_inlined=True):
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(j_func(*args), expected)
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        fir.blocks = fir.blocks
        self.assertEqual(len(fir.blocks), block_count)
        if expects_inlined:
            for block in fir.blocks.values():
                getattrs = list(block.find_exprs('getattr'))
                self.assertFalse(getattrs)
        else:
            allgetattrs = []
            for block in fir.blocks.values():
                allgetattrs += list(block.find_exprs('getattr'))
            self.assertTrue(allgetattrs)

    def test_overload_method_default_args_always(self):
        Dummy, DummyType = self.make_dummy_type()

        @overload_method(DummyType, 'inline_method', inline='always')
        def _get_inlined_method(obj, val=None, val2=None):

            def get(obj, val=None, val2=None):
                return ('THIS IS INLINED', val, val2)
            return get

        def foo(obj):
            return (obj.inline_method(123), obj.inline_method(val2=321))
        self.check_method(test_impl=foo, args=[Dummy()], expected=(('THIS IS INLINED', 123, None), ('THIS IS INLINED', None, 321)), block_count=1)

    def make_overload_method_test(self, costmodel, should_inline):

        def costmodel(*args):
            return should_inline
        Dummy, DummyType = self.make_dummy_type()

        @overload_method(DummyType, 'inline_method', inline=costmodel)
        def _get_inlined_method(obj, val):

            def get(obj, val):
                return ('THIS IS INLINED!!!', val)
            return get

        def foo(obj):
            return obj.inline_method(123)
        self.check_method(test_impl=foo, args=[Dummy()], expected=('THIS IS INLINED!!!', 123), block_count=1, expects_inlined=should_inline)

    def test_overload_method_cost_driven_always(self):
        self.make_overload_method_test(costmodel='always', should_inline=True)

    def test_overload_method_cost_driven_never(self):
        self.make_overload_method_test(costmodel='never', should_inline=False)

    def test_overload_method_cost_driven_must_inline(self):
        self.make_overload_method_test(costmodel=lambda *args: True, should_inline=True)

    def test_overload_method_cost_driven_no_inline(self):
        self.make_overload_method_test(costmodel=lambda *args: False, should_inline=False)

    def make_overload_attribute_test(self, costmodel, should_inline):
        Dummy, DummyType = self.make_dummy_type()

        @overload_attribute(DummyType, 'inlineme', inline=costmodel)
        def _get_inlineme(obj):

            def get(obj):
                return 'MY INLINED ATTRS'
            return get

        def foo(obj):
            return obj.inlineme
        self.check_getattr(test_impl=foo, args=[Dummy()], expected='MY INLINED ATTRS', block_count=1, expects_inlined=should_inline)

    def test_overload_attribute_always(self):
        self.make_overload_attribute_test(costmodel='always', should_inline=True)

    def test_overload_attribute_never(self):
        self.make_overload_attribute_test(costmodel='never', should_inline=False)

    def test_overload_attribute_costmodel_must_inline(self):
        self.make_overload_attribute_test(costmodel=lambda *args: True, should_inline=True)

    def test_overload_attribute_costmodel_no_inline(self):
        self.make_overload_attribute_test(costmodel=lambda *args: False, should_inline=False)