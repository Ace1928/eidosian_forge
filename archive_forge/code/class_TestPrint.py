from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import subprocess
import sys
import unittest
from numba import cuda
from numba import cuda
from numba import cuda
from numba import cuda
from numba import cuda
import numpy as np
class TestPrint(CUDATestCase):

    def run_code(self, code):
        """Runs code in a subprocess and returns the captured output"""
        cmd = [sys.executable, '-c', code]
        cp = subprocess.run(cmd, timeout=60, capture_output=True, check=True)
        return (cp.stdout.decode(), cp.stderr.decode())

    def test_cuhello(self):
        output, _ = self.run_code(cuhello_usecase)
        actual = [line.strip() for line in output.splitlines()]
        expected = ['-42'] * 6 + ['%d 999' % i for i in range(6)]
        self.assertEqual(sorted(actual), expected)

    def test_printfloat(self):
        output, _ = self.run_code(printfloat_usecase)
        expected_cases = ['0 23 34.750000 321', '0 23 34.75 321']
        self.assertIn(output.strip(), expected_cases)

    def test_printempty(self):
        output, _ = self.run_code(printempty_usecase)
        self.assertEqual(output.strip(), '')

    def test_string(self):
        output, _ = self.run_code(printstring_usecase)
        lines = [line.strip() for line in output.splitlines(True)]
        expected = ['%d hop! 999' % i for i in range(3)]
        self.assertEqual(sorted(lines), expected)

    @skip_on_cudasim('cudasim can print unlimited output')
    def test_too_many_args(self):
        output, errors = self.run_code(print_too_many_usecase)
        expected_fmt_string = ' '.join(['%lld' for _ in range(33)])
        self.assertIn(expected_fmt_string, output)
        warn_msg = 'CUDA print() cannot print more than 32 items. The raw format string will be emitted by the kernel instead.'
        self.assertIn(warn_msg, errors)