from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
@dataclass
class FastMathCriterion:
    fast_expected: List[str] = field(default_factory=list)
    fast_unexpected: List[str] = field(default_factory=list)
    prec_expected: List[str] = field(default_factory=list)
    prec_unexpected: List[str] = field(default_factory=list)

    def check(self, test: CUDATestCase, fast: str, prec: str):
        test.assertTrue(all((i in fast for i in self.fast_expected)))
        test.assertTrue(all((i not in fast for i in self.fast_unexpected)))
        test.assertTrue(all((i in prec for i in self.prec_expected)))
        test.assertTrue(all((i not in prec for i in self.prec_unexpected)))