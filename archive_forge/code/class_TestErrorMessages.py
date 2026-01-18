import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
class TestErrorMessages(unittest.TestCase):

    def test_specific_error(self):
        given_reason = 'specific_reason'

        def foo():
            pass

        @overload(foo)
        def ol_foo():
            raise errors.NumbaValueError(given_reason)

        @njit
        def call_foo():
            foo()
        with self.assertRaises(errors.TypingError) as raises:
            call_foo()
        excstr = str(raises.exception)
        self.assertIn(error_reasons['specific_error'].splitlines()[0], excstr)
        self.assertIn(given_reason, excstr)

    def test_no_match_error(self):

        def foo():
            pass

        @overload(foo)
        def ol_foo():
            return None

        @njit
        def call_foo():
            foo()
        with self.assertRaises(errors.TypingError) as raises:
            call_foo()
        excstr = str(raises.exception)
        self.assertIn('No match', excstr)

    @skip_unless_scipy
    def test_error_function_source_is_correct(self):
        """ Checks that the reported source location for an overload is the
        overload implementation source, not the actual function source from the
        target library."""

        @njit
        def foo():
            np.linalg.svd('chars')
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn(error_reasons['specific_error'].splitlines()[0], excstr)
        expected_file = os.path.join('numba', 'np', 'linalg.py')
        expected = f"Overload in function 'svd_impl': File: {expected_file}:"
        self.assertIn(expected.format(expected_file), excstr)

    def test_concrete_template_source(self):

        @njit
        def foo():
            return 'a' + 1
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn("Overload of function 'add'", excstr)
        self.assertIn('No match.', excstr)

    def test_abstract_template_source(self):

        @njit
        def foo():
            return len(1)
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn("Overload of function 'len'", excstr)

    def test_callable_template_source(self):

        @njit
        def foo():
            return np.angle(None)
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn('No implementation of function Function(<function angle', excstr)

    def test_overloadfunction_template_source(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            pass

        @njit
        def foo():
            return bar(1)
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertNotIn('<numerous>', excstr)
        expected_file = os.path.join('numba', 'tests', 'test_errorhandling.py')
        expected_ol = f"Overload of function 'bar': File: {expected_file}:"
        self.assertIn(expected_ol.format(expected_file), excstr)
        self.assertIn('No match.', excstr)

    def test_intrinsic_template_source(self):
        given_reason1 = 'x must be literal'
        given_reason2 = 'array.ndim must be 1'

        @intrinsic
        def myintrin(typingctx, x, arr):
            if not isinstance(x, types.IntegerLiteral):
                raise errors.RequireLiteralValue(given_reason1)
            if arr.ndim != 1:
                raise errors.NumbaValueError(given_reason2)
            sig = types.intp(x, arr)

            def codegen(context, builder, signature, args):
                pass
            return (sig, codegen)

        @njit
        def call_intrin():
            arr = np.zeros((2, 2))
            myintrin(1, arr)
        with self.assertRaises(errors.TypingError) as raises:
            call_intrin()
        excstr = str(raises.exception)
        self.assertIn(error_reasons['specific_error'].splitlines()[0], excstr)
        self.assertIn(given_reason1, excstr)
        self.assertIn(given_reason2, excstr)
        self.assertIn('Intrinsic in function', excstr)

    def test_overloadmethod_template_source(self):

        @overload_method(types.UnicodeType, 'isnonsense')
        def ol_unicode_isnonsense(self):
            pass

        @njit
        def foo():
            'abc'.isnonsense()
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn("Overload of function 'ol_unicode_isnonsense'", excstr)

    def test_overloadattribute_template_source(self):

        @overload_attribute(types.UnicodeType, 'isnonsense')
        def ol_unicode_isnonsense(self):
            pass

        @njit
        def foo():
            'abc'.isnonsense
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn("Overload of function 'ol_unicode_isnonsense'", excstr)

    def test_external_function_pointer_template_source(self):
        from numba.tests.ctypes_usecases import c_cos

        @njit
        def foo():
            c_cos('a')
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn("Type Restricted Function in function 'unknown'", excstr)

    @skip_unless_cffi
    def test_cffi_function_pointer_template_source(self):
        from numba.tests import cffi_usecases as mod
        mod.init()
        func = mod.cffi_cos

        @njit
        def foo():
            func('a')
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn("Type Restricted Function in function 'unknown'", excstr)

    def test_missing_source(self):

        @structref.register
        class ParticleType(types.StructRef):
            pass

        class Particle(structref.StructRefProxy):

            def __new__(cls, pos, mass):
                return structref.StructRefProxy.__new__(cls, pos)
        structref.define_proxy(Particle, ParticleType, ['pos', 'mass'])
        with self.assertRaises(errors.TypingError) as raises:
            Particle(pos=1, mass=2)
        excstr = str(raises.exception)
        self.assertIn("missing a required argument: 'mass'", excstr)