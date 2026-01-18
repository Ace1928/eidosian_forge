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
class TestLiteralTupleInterpretation(MemoryLeakMixin, TestCase):

    def check(self, func, var):
        cres = func.overloads[func.signatures[0]]
        ty = cres.fndesc.typemap[var]
        self.assertTrue(isinstance(ty, types.Tuple))
        for subty in ty:
            self.assertTrue(isinstance(subty, types.Literal), 'non literal')

    def test_homogeneous_literal(self):

        @njit
        def foo():
            x = (1, 2, 3)
            return x[1]
        self.assertEqual(foo(), foo.py_func())
        self.check(foo, 'x')

    def test_heterogeneous_literal(self):

        @njit
        def foo():
            x = (1, 2, 3, 'a')
            return x[3]
        self.assertEqual(foo(), foo.py_func())
        self.check(foo, 'x')

    def test_non_literal(self):

        @njit
        def foo():
            x = (1, 2, 3, 'a', 1j)
            return x[4]
        self.assertEqual(foo(), foo.py_func())
        with self.assertRaises(AssertionError) as e:
            self.check(foo, 'x')
        self.assertIn('non literal', str(e.exception))