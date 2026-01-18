from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
@skip_on_cudasim('Compilation unsupported in the simulator')
class TestCompileToPTX(unittest.TestCase):

    def test_global_kernel(self):

        def f(r, x, y):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = x[i] + y[i]
        args = (float32[:], float32[:], float32[:])
        ptx, resty = compile_ptx(f, args)
        self.assertNotIn('func_retval', ptx)
        self.assertNotIn('.visible .func', ptx)
        self.assertIn('.visible .entry', ptx)
        self.assertEqual(resty, void)

    def test_device_function(self):

        def add(x, y):
            return x + y
        args = (float32, float32)
        ptx, resty = compile_ptx(add, args, device=True)
        self.assertIn('func_retval', ptx)
        self.assertIn('.visible .func', ptx)
        self.assertNotIn('.visible .entry', ptx)
        self.assertEqual(resty, float32)
        sig_int32 = int32(int32, int32)
        ptx, resty = compile_ptx(add, sig_int32, device=True)
        self.assertEqual(resty, int32)
        sig_int16 = int16(int16, int16)
        ptx, resty = compile_ptx(add, sig_int16, device=True)
        self.assertEqual(resty, int16)
        sig_string = 'uint32(uint32, uint32)'
        ptx, resty = compile_ptx(add, sig_string, device=True)
        self.assertEqual(resty, uint32)

    def test_fastmath(self):

        def f(x, y, z, d):
            return sqrt((x * y + z) / d)
        args = (float32, float32, float32, float32)
        ptx, resty = compile_ptx(f, args, device=True)
        self.assertIn('fma.rn.f32', ptx)
        self.assertIn('div.rn.f32', ptx)
        self.assertIn('sqrt.rn.f32', ptx)
        ptx, resty = compile_ptx(f, args, device=True, fastmath=True)
        self.assertIn('fma.rn.ftz.f32', ptx)
        self.assertIn('div.approx.ftz.f32', ptx)
        self.assertIn('sqrt.approx.ftz.f32', ptx)

    def check_debug_info(self, ptx):
        self.assertRegex(ptx, '\\.section\\s+\\.debug_info')
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    def test_device_function_with_debug(self):

        def f():
            pass
        ptx, resty = compile_ptx(f, (), device=True, debug=True)
        self.check_debug_info(ptx)

    def test_kernel_with_debug(self):

        def f():
            pass
        ptx, resty = compile_ptx(f, (), debug=True)
        self.check_debug_info(ptx)

    def check_line_info(self, ptx):
        self.assertRegex(ptx, '\\.file.*test_compiler.py"')

    def test_device_function_with_line_info(self):

        def f():
            pass
        ptx, resty = compile_ptx(f, (), device=True, lineinfo=True)
        self.check_line_info(ptx)

    def test_kernel_with_line_info(self):

        def f():
            pass
        ptx, resty = compile_ptx(f, (), lineinfo=True)
        self.check_line_info(ptx)

    def test_non_void_return_type(self):

        def f(x, y):
            return x[0] + y[0]
        with self.assertRaisesRegex(TypeError, 'must have void return type'):
            compile_ptx(f, (uint32[::1], uint32[::1]))

    def test_c_abi_disallowed_for_kernel(self):

        def f(x, y):
            return x + y
        with self.assertRaisesRegex(NotImplementedError, 'The C ABI is not supported for kernels'):
            compile_ptx(f, (int32, int32), abi='c')

    def test_unsupported_abi(self):

        def f(x, y):
            return x + y
        with self.assertRaisesRegex(NotImplementedError, 'Unsupported ABI: fastcall'):
            compile_ptx(f, (int32, int32), abi='fastcall')

    def test_c_abi_device_function(self):

        def f(x, y):
            return x + y
        ptx, resty = compile_ptx(f, int32(int32, int32), device=True, abi='c')
        self.assertNotIn(ptx, 'param_2')
        self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b32\\s+func_retval0\\)\\s+f\\(')
        ptx, resty = compile_ptx(f, int64(int64, int64), device=True, abi='c')
        self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b64')

    def test_c_abi_device_function_module_scope(self):
        ptx, resty = compile_ptx(f_module, int32(int32, int32), device=True, abi='c')
        self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b32\\s+func_retval0\\)\\s+f_module\\(')

    def test_c_abi_with_abi_name(self):
        abi_info = {'abi_name': '_Z4funcii'}
        ptx, resty = compile_ptx(f_module, int32(int32, int32), device=True, abi='c', abi_info=abi_info)
        self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b32\\s+func_retval0\\)\\s+_Z4funcii\\(')