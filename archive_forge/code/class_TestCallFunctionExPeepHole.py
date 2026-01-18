import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
class TestCallFunctionExPeepHole(MemoryLeakMixin, TestCase):
    """
    gh #7812

    Tests that check a peephole optimization for Function calls
    in Python 3.10. The bytecode changes when
    (n_args / 2) + n_kws > 15, which moves the arguments from
    the stack into a tuple and dictionary.

    This peephole optimization updates the IR to use the original format.
    There are different paths when n_args > 30 and n_args <= 30 and when
    n_kws > 15 and n_kws <= 15.
    """
    THRESHOLD_ARGS = 31
    THRESHOLD_KWS = 16

    def gen_func(self, n_args, n_kws):
        """
            Generates a function that calls sum_jit_func
            with the desired number of args and kws.
        """
        param_list = [f'arg{i}' for i in range(n_args + n_kws)]
        args_list = []
        for i in range(n_args + n_kws):
            if i % 5 == 0:
                arg_val = f'pow(arg{i}, 2)'
            else:
                arg_val = f'arg{i}'
            args_list.append(arg_val)
        total_params = ', '.join(param_list)
        func_text = f'def impl({total_params}):\n'
        func_text += '    return sum_jit_func(\n'
        for i in range(n_args):
            func_text += f'        {args_list[i]},\n'
        for i in range(n_args, n_args + n_kws):
            func_text += f'        {param_list[i]}={args_list[i]},\n'
        func_text += '    )\n'
        local_vars = {}
        exec(func_text, {'sum_jit_func': sum_jit_func}, local_vars)
        return local_vars['impl']

    @skip_unless_py10_or_later
    def test_all_args(self):
        """
        Tests calling a function when n_args > 30 and
        n_kws = 0. This shouldn't use the peephole, but
        it should still succeed.
        """
        total_args = [i for i in range(self.THRESHOLD_ARGS)]
        f = self.gen_func(self.THRESHOLD_ARGS, 0)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_all_kws(self):
        """
        Tests calling a function when n_kws > 15 and
        n_args = 0.
        """
        total_args = [i for i in range(self.THRESHOLD_KWS)]
        f = self.gen_func(0, self.THRESHOLD_KWS)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_small_args_small_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args <= 30 and n_kws <= 15
        """
        used_args = self.THRESHOLD_ARGS - 1
        used_kws = self.THRESHOLD_KWS - 1
        total_args = [i for i in range(used_args + used_kws)]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_small_args_large_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args <= 30 and n_kws > 15
        """
        used_args = self.THRESHOLD_ARGS - 1
        used_kws = self.THRESHOLD_KWS
        total_args = [i for i in range(used_args + used_kws)]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_args_small_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args > 30 and n_kws <= 15
        """
        used_args = self.THRESHOLD_ARGS
        used_kws = self.THRESHOLD_KWS - 1
        total_args = [i for i in range(used_args + used_kws)]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_args_large_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args > 30 and n_kws > 15
        """
        used_args = self.THRESHOLD_ARGS
        used_kws = self.THRESHOLD_KWS
        total_args = [i for i in range(used_args + used_kws)]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_kws_objmode(self):
        """
        Tests calling an objectmode function with > 15 return values.
        """

        def py_func():
            return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        @njit
        def objmode_func():
            """
            Wrapper to call py_func from objmode. This tests
            large kws with objmode. If the definition for the
            call is not properly updated this test will fail.
            """
            with objmode(a='int64', b='int64', c='int64', d='int64', e='int64', f='int64', g='int64', h='int64', i='int64', j='int64', k='int64', l='int64', m='int64', n='int64', o='int64', p='int64'):
                a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p = py_func()
            return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p
        a = sum(list(py_func()))
        b = objmode_func()
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_args_inline_controlflow(self):
        """
        Tests generating large args when one of the inputs
        has inlined controlflow.
        """

        def inline_func(flag):
            return sum_jit_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 if flag else 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, arg41=1)
        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)(False)
        self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))

    @skip_unless_py10_or_later
    def test_large_args_noninlined_controlflow(self):
        """
        Tests generating large args when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """

        def inline_func(flag):
            a_val = 1 if flag else 2
            return sum_jit_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, a_val, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, arg41=1)
        py_func = inline_func
        cfunc = njit()(inline_func)
        a = py_func(False)
        b = cfunc(False)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_all_args_inline_controlflow(self):
        """
        Tests generating only large args when one of the inputs
        has inlined controlflow. This requires a special check
        inside peep_hole_call_function_ex_to_call_function_kw
        because it usually only handles varkwargs.
        """

        def inline_func(flag):
            return sum_jit_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 if flag else 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)(False)
        self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))

    @skip_unless_py10_or_later
    def test_all_args_noninlined_controlflow(self):
        """
        Tests generating large args when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """

        def inline_func(flag):
            a_val = 1 if flag else 2
            return sum_jit_func(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, a_val, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        py_func = inline_func
        cfunc = njit()(inline_func)
        a = py_func(False)
        b = cfunc(False)
        self.assertEqual(a, b)

    @skip_unless_py10_or_later
    def test_large_kws_inline_controlflow(self):
        """
        Tests generating large kws when one of the inputs
        has inlined controlflow.
        """

        def inline_func(flag):
            return sum_jit_func(arg0=1, arg1=1, arg2=1, arg3=1, arg4=1, arg5=1, arg6=1, arg7=1, arg8=1, arg9=1, arg10=1, arg11=1, arg12=1, arg13=1, arg14=1, arg15=1 if flag else 2)
        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)(False)
        self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))

    @skip_unless_py10_or_later
    def test_large_kws_noninlined_controlflow(self):
        """
        Tests generating large kws when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """

        def inline_func(flag):
            a_val = 1 if flag else 2
            return sum_jit_func(arg0=1, arg1=1, arg2=1, arg3=1, arg4=1, arg5=1, arg6=1, arg7=1, arg8=1, arg9=1, arg10=1, arg11=1, arg12=1, arg13=1, arg14=1, arg15=a_val)
        py_func = inline_func
        cfunc = njit()(inline_func)
        a = py_func(False)
        b = cfunc(False)
        self.assertEqual(a, b)