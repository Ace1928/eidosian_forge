import numpy as np
from numba import cuda, float32, int32, void
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from .extensions_usecases import test_struct_model_type
class TestSharedMemoryCreation(CUDATestCase):

    def getarg(self):
        return np.array(100, dtype=np.float32, ndmin=1)

    def getarg2(self):
        return self.getarg().reshape(1, 1)

    def test_global_constants(self):
        udt = cuda.jit((float32[:],))(udt_global_constants)
        udt[1, 1](self.getarg())

    def test_global_build_tuple(self):
        udt = cuda.jit((float32[:, :],))(udt_global_build_tuple)
        udt[1, 1](self.getarg2())

    @skip_on_cudasim('Simulator does not prohibit lists for shared array shape')
    def test_global_build_list(self):
        with self.assertRaises(TypingError) as raises:
            cuda.jit((float32[:, :],))(udt_global_build_list)
        self.assertIn('No implementation of function Function(<function shared.array', str(raises.exception))
        self.assertIn('found for signature:\n \n >>> array(shape=list(int64)<iv=[5, 6]>, dtype=class(float32)', str(raises.exception))

    def test_global_constant_tuple(self):
        udt = cuda.jit((float32[:, :],))(udt_global_constant_tuple)
        udt[1, 1](self.getarg2())

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_1(self):
        with self.assertRaises(TypingError) as raises:
            cuda.jit((float32[:],))(udt_invalid_1)
        self.assertIn('No implementation of function Function(<function shared.array', str(raises.exception))
        self.assertIn('found for signature:\n \n >>> array(shape=float32, dtype=class(float32))', str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_2(self):
        with self.assertRaises(TypingError) as raises:
            cuda.jit((float32[:, :],))(udt_invalid_2)
        self.assertIn('No implementation of function Function(<function shared.array', str(raises.exception))
        self.assertIn('found for signature:\n \n >>> array(shape=Tuple(Literal[int](1), array(float32, 1d, A)), dtype=class(float32))', str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_3(self):
        with self.assertRaises(TypingError) as raises:
            cuda.jit((int32[:],))(udt_invalid_1)
        self.assertIn('No implementation of function Function(<function shared.array', str(raises.exception))
        self.assertIn('found for signature:\n \n >>> array(shape=int32, dtype=class(float32))', str(raises.exception))

    @skip_on_cudasim("Can't check for constants in simulator")
    def test_invalid_4(self):
        with self.assertRaises(TypingError) as raises:
            cuda.jit((int32[:],))(udt_invalid_3)
        self.assertIn('No implementation of function Function(<function shared.array', str(raises.exception))
        self.assertIn('found for signature:\n \n >>> array(shape=Tuple(Literal[int](1), int32), dtype=class(float32))', str(raises.exception))

    def check_dtype(self, f, dtype):
        annotation = next(iter(f.overloads.values()))._type_annotation
        l_dtype = annotation.typemap['s'].dtype
        self.assertEqual(l_dtype, dtype)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_numba_dtype(self):

        @cuda.jit(void(int32[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype=int32)
            s[0] = x[0]
            x[0] = s[0]
        self.check_dtype(f, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_numpy_dtype(self):

        @cuda.jit(void(int32[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype=np.int32)
            s[0] = x[0]
            x[0] = s[0]
        self.check_dtype(f, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_string_dtype(self):

        @cuda.jit(void(int32[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype='int32')
            s[0] = x[0]
            x[0] = s[0]
        self.check_dtype(f, int32)

    @skip_on_cudasim("Can't check typing in simulator")
    def test_invalid_string_dtype(self):
        re = ".*Invalid NumPy dtype specified: 'int33'.*"
        with self.assertRaisesRegex(TypingError, re):

            @cuda.jit(void(int32[::1]))
            def f(x):
                s = cuda.shared.array(10, dtype='int33')
                s[0] = x[0]
                x[0] = s[0]

    @skip_on_cudasim("Can't check typing in simulator")
    def test_type_with_struct_data_model(self):

        @cuda.jit(void(test_struct_model_type[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype=test_struct_model_type)
            s[0] = x[0]
            x[0] = s[0]
        self.check_dtype(f, test_struct_model_type)