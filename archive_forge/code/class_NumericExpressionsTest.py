from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class NumericExpressionsTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        ppc = pp.pyparsing_common
        real = ppc.real().setParseAction(None)
        sci_real = ppc.sci_real().setParseAction(None)
        signed_integer = ppc.signed_integer().setParseAction(None)
        from itertools import product

        def make_tests():
            leading_sign = ['+', '-', '']
            leading_digit = ['0', '']
            dot = ['.', '']
            decimal_digit = ['1', '']
            e = ['e', 'E', '']
            e_sign = ['+', '-', '']
            e_int = ['22', '']
            stray = ['9', '.', '']
            seen = set()
            seen.add('')
            for parts in product(leading_sign, stray, leading_digit, dot, decimal_digit, stray, e, e_sign, e_int, stray):
                parts_str = ''.join(parts).strip()
                if parts_str in seen:
                    continue
                seen.add(parts_str)
                yield parts_str
            print_(len(seen) - 1, 'tests produced')
        valid_ints = set()
        valid_reals = set()
        valid_sci_reals = set()
        invalid_ints = set()
        invalid_reals = set()
        invalid_sci_reals = set()
        for test_str in make_tests():
            if '.' in test_str or 'e' in test_str.lower():
                try:
                    float(test_str)
                except ValueError:
                    invalid_sci_reals.add(test_str)
                    if 'e' not in test_str.lower():
                        invalid_reals.add(test_str)
                else:
                    valid_sci_reals.add(test_str)
                    if 'e' not in test_str.lower():
                        valid_reals.add(test_str)
            try:
                int(test_str)
            except ValueError:
                invalid_ints.add(test_str)
            else:
                valid_ints.add(test_str)
        all_pass = True
        suppress_results = {'printResults': False}
        for expr, tests, is_fail, fn in zip([real, sci_real, signed_integer] * 2, [valid_reals, valid_sci_reals, valid_ints, invalid_reals, invalid_sci_reals, invalid_ints], [False, False, False, True, True, True], [float, float, int] * 2):
            success = True
            for t in tests:
                if expr.matches(t, parseAll=True):
                    if is_fail:
                        print_(t, 'should fail but did not')
                        success = False
                elif not is_fail:
                    print_(t, 'should not fail but did')
                    success = False
            print_(expr, ('FAIL', 'PASS')[success], '{1}valid tests ({0})'.format(len(tests), 'in' if is_fail else ''))
            all_pass = all_pass and success
        self.assertTrue(all_pass, 'failed one or more numeric tests')