import functools
import sys
import types
import warnings
import unittest
class TestObsoleteFunctions(unittest.TestCase):

    class MyTestSuite(unittest.TestSuite):
        pass

    class MyTestCase(unittest.TestCase):

        def check_1(self):
            pass

        def check_2(self):
            pass

        def test(self):
            pass

    @staticmethod
    def reverse_three_way_cmp(a, b):
        return unittest.util.three_way_cmp(b, a)

    def test_getTestCaseNames(self):
        with self.assertWarns(DeprecationWarning) as w:
            tests = unittest.getTestCaseNames(self.MyTestCase, prefix='check', sortUsing=self.reverse_three_way_cmp, testNamePatterns=None)
        self.assertEqual(w.filename, __file__)
        self.assertEqual(tests, ['check_2', 'check_1'])

    def test_makeSuite(self):
        with self.assertWarns(DeprecationWarning) as w:
            suite = unittest.makeSuite(self.MyTestCase, prefix='check', sortUsing=self.reverse_three_way_cmp, suiteClass=self.MyTestSuite)
        self.assertEqual(w.filename, __file__)
        self.assertIsInstance(suite, self.MyTestSuite)
        expected = self.MyTestSuite([self.MyTestCase('check_2'), self.MyTestCase('check_1')])
        self.assertEqual(suite, expected)

    def test_findTestCases(self):
        m = types.ModuleType('m')
        m.testcase_1 = self.MyTestCase
        with self.assertWarns(DeprecationWarning) as w:
            suite = unittest.findTestCases(m, prefix='check', sortUsing=self.reverse_three_way_cmp, suiteClass=self.MyTestSuite)
        self.assertEqual(w.filename, __file__)
        self.assertIsInstance(suite, self.MyTestSuite)
        expected = [self.MyTestSuite([self.MyTestCase('check_2'), self.MyTestCase('check_1')])]
        self.assertEqual(list(suite), expected)