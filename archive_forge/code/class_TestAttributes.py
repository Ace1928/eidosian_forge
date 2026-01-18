import unittest
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
class TestAttributes(TestCase):

    def test_setattr(self):
        pyfunc = setattr_usecase
        cfunc = jit((types.pyobject, types.int32), forceobj=True)(pyfunc)
        c = C()
        cfunc(c, 123)
        self.assertEqual(c.x, 123)

    def test_setattr_attribute_error(self):
        pyfunc = setattr_usecase
        cfunc = jit((types.pyobject, types.int32), forceobj=True)(pyfunc)
        with self.assertRaises(AttributeError):
            cfunc(object(), 123)

    def test_delattr(self):
        pyfunc = delattr_usecase
        cfunc = jit((types.pyobject,), forceobj=True)(pyfunc)
        c = C()
        c.x = 123
        cfunc(c)
        with self.assertRaises(AttributeError):
            c.x

    def test_delattr_attribute_error(self):
        pyfunc = delattr_usecase
        cfunc = jit((types.pyobject,), forceobj=True)(pyfunc)
        with self.assertRaises(AttributeError):
            cfunc(C())