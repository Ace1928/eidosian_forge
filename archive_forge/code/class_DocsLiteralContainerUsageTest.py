import unittest
from numba.tests.support import captured_stdout
from numba import typed
class DocsLiteralContainerUsageTest(unittest.TestCase):

    def test_ex_literal_dict_compile_time_consts(self):
        with captured_stdout():
            import numpy as np
            from numba import njit, types
            from numba.extending import overload

            def specialize(x):
                pass

            @overload(specialize)
            def ol_specialize(x):
                ld = x.literal_value
                const_expr = []
                for k, v in ld.items():
                    if isinstance(v, types.Literal):
                        lv = v.literal_value
                        if lv == 'cat':
                            const_expr.append('Meow!')
                        elif lv == 'dog':
                            const_expr.append('Woof!')
                        elif isinstance(lv, int):
                            const_expr.append(k.literal_value * lv)
                    else:
                        const_expr.append('Array(dim={dim}'.format(dim=v.ndim))
                const_strings = tuple(const_expr)

                def impl(x):
                    return const_strings
                return impl

            @njit
            def foo():
                pets_ints_and_array = {'a': 1, 'b': 2, 'c': 'cat', 'd': 'dog', 'e': np.ones(5)}
                return specialize(pets_ints_and_array)
            result = foo()
            print(result)
        self.assertEqual(result, ('a', 'bb', 'Meow!', 'Woof!', 'Array(dim=1'))

    def test_ex_initial_value_dict_compile_time_consts(self):
        with captured_stdout():
            from numba import njit, literally
            from numba.extending import overload

            def specialize(x):
                pass

            @overload(specialize)
            def ol_specialize(x):
                iv = x.initial_value
                if iv is None:
                    return lambda x: literally(x)
                assert iv == {'a': 1, 'b': 2, 'c': 3}
                return lambda x: literally(x)

            @njit
            def foo():
                d = {'a': 1, 'b': 2, 'c': 3}
                d['c'] = 20
                d['d'] = 30
                return specialize(d)
            result = foo()
            print(result)
        expected = typed.Dict()
        for k, v in {'a': 1, 'b': 2, 'c': 20, 'd': 30}.items():
            expected[k] = v
        self.assertEqual(result, expected)

    def test_ex_literal_list(self):
        with captured_stdout():
            from numba import njit
            from numba.extending import overload

            def specialize(x):
                pass

            @overload(specialize)
            def ol_specialize(x):
                l = x.literal_value
                const_expr = []
                for v in l:
                    const_expr.append(str(v))
                const_strings = tuple(const_expr)

                def impl(x):
                    return const_strings
                return impl

            @njit
            def foo():
                const_list = ['a', 10, 1j, ['another', 'list']]
                return specialize(const_list)
            result = foo()
            print(result)
        expected = ('Literal[str](a)', 'Literal[int](10)', 'complex128', "list(unicode_type)<iv=['another', 'list']>")
        self.assertEqual(result, expected)

    def test_ex_initial_value_list_compile_time_consts(self):
        with captured_stdout():
            from numba import njit, literally
            from numba.extending import overload

            def specialize(x):
                pass

            @overload(specialize)
            def ol_specialize(x):
                iv = x.initial_value
                if iv is None:
                    return lambda x: literally(x)
                assert iv == [1, 2, 3]
                return lambda x: x

            @njit
            def foo():
                l = [1, 2, 3]
                l[2] = 20
                l.append(30)
                return specialize(l)
            result = foo()
            print(result)
        expected = [1, 2, 20, 30]
        self.assertEqual(result, expected)