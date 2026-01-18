from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
class AsyncTest(TestCase):

    def test_should_be_async(self):
        self.assertFalse(_should_be_async('False'))
        self.assertTrue(_should_be_async('await bar()'))
        self.assertTrue(_should_be_async('x = await bar()'))
        self.assertFalse(_should_be_async(dedent('\n            async def awaitable():\n                pass\n        ')))

    def _get_top_level_cases(self):
        test_cases = []
        test_cases.append(('basic', '{val}'))
        test_cases.append(('if', dedent('\n        if True:\n            {val}\n        ')))
        test_cases.append(('while', dedent('\n        while True:\n            {val}\n            break\n        ')))
        test_cases.append(('try', dedent('\n        try:\n            {val}\n        except:\n            pass\n        ')))
        test_cases.append(('except', dedent('\n        try:\n            pass\n        except:\n            {val}\n        ')))
        test_cases.append(('finally', dedent('\n        try:\n            pass\n        except:\n            pass\n        finally:\n            {val}\n        ')))
        test_cases.append(('for', dedent('\n        for _ in range(4):\n            {val}\n        ')))
        test_cases.append(('nested', dedent('\n        if True:\n            while True:\n                {val}\n                break\n        ')))
        test_cases.append(('deep-nested', dedent('\n        if True:\n            while True:\n                break\n                for x in range(3):\n                    if True:\n                        while True:\n                            for x in range(3):\n                                {val}\n        ')))
        return test_cases

    def _get_ry_syntax_errors(self):
        test_cases = []
        test_cases.append(('class', dedent('\n        class V:\n            {val}\n        ')))
        test_cases.append(('nested-class', dedent('\n        class V:\n            class C:\n                {val}\n        ')))
        return test_cases

    def test_top_level_return_error(self):
        tl_err_test_cases = self._get_top_level_cases()
        tl_err_test_cases.extend(self._get_ry_syntax_errors())
        vals = ('return', 'yield', 'yield from (_ for _ in range(3))', dedent('\n                    def f():\n                        pass\n                    return\n                    '))
        for test_name, test_case in tl_err_test_cases:
            with self.subTest((test_name, 'pass')):
                iprc(test_case.format(val='pass'))
            for val in vals:
                with self.subTest((test_name, val)):
                    msg = 'Syntax error not raised for %s, %s' % (test_name, val)
                    with self.assertRaises(SyntaxError, msg=msg):
                        iprc(test_case.format(val=val))

    def test_in_func_no_error(self):
        func_contexts = []
        func_contexts.append(('func', False, dedent('\n        def f():')))
        func_contexts.append(('method', False, dedent('\n        class MyClass:\n            def __init__(self):\n        ')))
        func_contexts.append(('async-func', True, dedent('\n        async def f():')))
        func_contexts.append(('async-method', True, dedent('\n        class MyClass:\n            async def f(self):')))
        func_contexts.append(('closure', False, dedent('\n        def f():\n            def g():\n        ')))

        def nest_case(context, case):
            lines = context.strip().splitlines()
            prefix_len = 0
            for c in lines[-1]:
                if c != ' ':
                    break
                prefix_len += 1
            indented_case = indent(case, ' ' * (prefix_len + 4))
            return context + '\n' + indented_case
        vals = ('return', 'yield', 'yield from (_ for _ in range(3))')
        success_tests = zip(self._get_top_level_cases(), repeat(False))
        failure_tests = zip(self._get_ry_syntax_errors(), repeat(True))
        tests = chain(success_tests, failure_tests)
        for context_name, async_func, context in func_contexts:
            for (test_name, test_case), should_fail in tests:
                nested_case = nest_case(context, test_case)
                for val in vals:
                    test_id = (context_name, test_name, val)
                    cell = nested_case.format(val=val)
                    with self.subTest(test_id):
                        if should_fail:
                            msg = 'SyntaxError not raised for %s' % str(test_id)
                            with self.assertRaises(SyntaxError, msg=msg):
                                iprc(cell)
                                print(cell)
                        else:
                            iprc(cell)

    def test_nonlocal(self):
        with self.assertRaises(SyntaxError):
            iprc('nonlocal x')
            iprc('\n            x = 1\n            def f():\n                nonlocal x\n                x = 10000\n                yield x\n            ')
            iprc('\n            def f():\n                def g():\n                    nonlocal x\n                    x = 10000\n                    yield x\n            ')
        iprc('\n        def f():\n            x = 20\n            def g():\n                nonlocal x\n                x = 10000\n                yield x\n        ')

    def test_execute(self):
        iprc('\n        import asyncio\n        await asyncio.sleep(0.001)\n        ')

    def test_autoawait(self):
        iprc('%autoawait False')
        iprc('%autoawait True')
        iprc('\n        from asyncio import sleep\n        await sleep(0.1)\n        ')

    def test_memory_error(self):
        """
        The pgen parser in 3.8 or before use to raise MemoryError on too many
        nested parens anymore"""
        iprc('(' * 200 + ')' * 200)

    @pytest.mark.xfail(reason='fail on curio 1.6 and before on Python 3.12')
    @pytest.mark.skip(reason='skip_without(curio) fails on 3.12 for now even with other skip so must uncond skip')
    def test_autoawait_curio(self):
        iprc('%autoawait curio')

    @skip_without('trio')
    def test_autoawait_trio(self):
        iprc('%autoawait trio')

    @skip_without('trio')
    def test_autoawait_trio_wrong_sleep(self):
        iprc('%autoawait trio')
        res = iprc_nr('\n        import asyncio\n        await asyncio.sleep(0)\n        ')
        with self.assertRaises(TypeError):
            res.raise_error()

    @skip_without('trio')
    def test_autoawait_asyncio_wrong_sleep(self):
        iprc('%autoawait asyncio')
        res = iprc_nr('\n        import trio\n        await trio.sleep(0)\n        ')
        with self.assertRaises(RuntimeError):
            res.raise_error()

    def tearDown(self):
        ip.loop_runner = 'asyncio'