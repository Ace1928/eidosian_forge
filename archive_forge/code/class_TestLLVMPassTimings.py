import unittest
from numba import njit
from numba.tests.support import TestCase, override_config
from numba.misc import llvm_pass_timings as lpt
class TestLLVMPassTimings(TestCase):

    def test_usage(self):

        @njit
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c
        with override_config('LLVM_PASS_TIMINGS', True):
            foo(10)
        md = foo.get_metadata(foo.signatures[0])
        timings = md['llvm_pass_timings']
        self.assertIsInstance(timings, lpt.PassTimingsCollection)
        text = str(timings)
        self.assertIn('Module passes (full optimization)', text)
        self.assertGreater(len(timings), 0)
        last = timings[-1]
        self.assertIsInstance(last, lpt.NamedTimings)
        self.assertIsInstance(last.name, str)
        self.assertIsInstance(last.timings, lpt.ProcessedPassTimings)

    def test_analyze(self):

        @njit
        def foo(n):
            c = 0
            for i in range(n):
                for j in range(i):
                    c += j
            return c
        with override_config('LLVM_PASS_TIMINGS', True):
            foo(10)
        md = foo.get_metadata(foo.signatures[0])
        timings_collection = md['llvm_pass_timings']
        self.assertIsInstance(timings_collection.get_total_time(), float)
        self.assertIsInstance(timings_collection.summary(), str)
        longest_first = timings_collection.list_longest_first()
        self.assertEqual(len(longest_first), len(timings_collection))
        last = longest_first[0].timings.get_total_time()
        for rec in longest_first[1:]:
            cur = rec.timings.get_total_time()
            self.assertGreaterEqual(last, cur)
            cur = last

    def test_parse_raw(self):
        timings1 = lpt.ProcessedPassTimings(timings_raw1)
        self.assertAlmostEqual(timings1.get_total_time(), 0.0001)
        self.assertIsInstance(timings1.summary(), str)
        timings2 = lpt.ProcessedPassTimings(timings_raw2)
        self.assertAlmostEqual(timings2.get_total_time(), 0.0001)
        self.assertIsInstance(timings2.summary(), str)