from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
class TestPickling(TestCase):
    """
    Pickling and unpickling should preserve type identity (singleton-ness)
    and the _code attribute.  This is only a requirement for types that
    can be part of function signatures.
    """

    def predefined_types(self):
        """
        Yield all predefined type instances
        """
        for ty in types.__dict__.values():
            if isinstance(ty, types.Type):
                yield ty

    def check_pickling(self, orig):
        pickled = pickle.dumps(orig, protocol=-1)
        ty = pickle.loads(pickled)
        self.assertIs(ty, orig)
        self.assertGreaterEqual(ty._code, 0)

    def test_predefined_types(self):
        tys = list(self.predefined_types())
        self.assertIn(types.int16, tys)
        for ty in tys:
            self.check_pickling(ty)

    def test_atomic_types(self):
        for unit in ('M', 'ms'):
            ty = types.NPDatetime(unit)
            self.check_pickling(ty)
            ty = types.NPTimedelta(unit)
            self.check_pickling(ty)

    def test_arrays(self):
        for ndim in (0, 1, 2):
            for layout in ('A', 'C', 'F'):
                ty = types.Array(types.int16, ndim, layout)
                self.check_pickling(ty)

    def test_records(self):
        recordtype = np.dtype([('a', np.float64), ('b', np.int32), ('c', np.complex64), ('d', (np.str_, 5))])
        ty = numpy_support.from_dtype(recordtype)
        self.check_pickling(ty)
        self.check_pickling(types.Array(ty, 1, 'A'))

    def test_optional(self):
        ty = types.Optional(types.int32)
        self.check_pickling(ty)

    def test_tuples(self):
        ty1 = types.UniTuple(types.int32, 3)
        self.check_pickling(ty1)
        ty2 = types.Tuple((types.int32, ty1))
        self.check_pickling(ty2)

    def test_namedtuples(self):
        ty1 = types.NamedUniTuple(types.intp, 2, Point)
        self.check_pickling(ty1)
        ty2 = types.NamedTuple((types.intp, types.float64), Point)
        self.check_pickling(ty2)

    def test_enums(self):
        ty1 = types.EnumMember(Color, types.int32)
        self.check_pickling(ty1)
        ty2 = types.EnumMember(Shake, types.int64)
        self.check_pickling(ty2)
        ty3 = types.IntEnumMember(Shape, types.int64)
        self.check_pickling(ty3)

    def test_lists(self):
        ty = types.List(types.int32)
        self.check_pickling(ty)

    def test_generator(self):
        cfunc = jit('(int32,)', nopython=True)(gen)
        sigs = list(cfunc.nopython_signatures)
        ty = sigs[0].return_type
        self.assertIsInstance(ty, types.Generator)
        self.check_pickling(ty)

    @unittest.expectedFailure
    def test_external_function_pointers(self):
        from numba.core.typing import ctypes_utils
        from numba.tests.ctypes_usecases import c_sin, c_cos
        for fnptr in (c_sin, c_cos):
            ty = ctypes_utils.make_function_type(fnptr)
            self.assertIsInstance(ty, types.ExternalFunctionPointer)
            self.check_pickling(ty)