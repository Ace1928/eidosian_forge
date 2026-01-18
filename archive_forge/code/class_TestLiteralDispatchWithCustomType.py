import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
class TestLiteralDispatchWithCustomType(TestCase):

    def make_dummy_type(self):

        class Dummy(object):

            def lit(self, a):
                return a

        class DummyType(types.Type):

            def __init__(self):
                super(DummyType, self).__init__(name='dummy')

        @register_model(DummyType)
        class DummyTypeModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                members = []
                super(DummyTypeModel, self).__init__(dmm, fe_type, members)

        @intrinsic
        def init_dummy(typingctx):

            def codegen(context, builder, signature, args):
                dummy = cgutils.create_struct_proxy(signature.return_type)(context, builder)
                return dummy._getvalue()
            sig = signature(DummyType())
            return (sig, codegen)

        @overload(Dummy)
        def dummy_overload():

            def ctor():
                return init_dummy()
            return ctor
        return (DummyType, Dummy)

    def test_overload_method(self):
        DummyType, Dummy = self.make_dummy_type()

        @overload_method(DummyType, 'lit')
        def lit_overload(self, a):

            def impl(self, a):
                return literally(a)
            return impl

        @njit
        def test_impl(a):
            d = Dummy()
            return d.lit(a)
        self.assertEqual(test_impl(5), 5)

        @njit
        def inside(a):
            return test_impl(a + 1)
        with self.assertRaises(errors.TypingError) as raises:
            inside(4)
        self.assertIn('Cannot request literal type.', str(raises.exception))