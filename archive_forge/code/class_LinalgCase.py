import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class LinalgCase:

    def __init__(self, name, a, b, tags=set()):
        """
        A bundle of arguments to be passed to a test case, with an identifying
        name, the operands a and b, and a set of tags to filter the tests
        """
        assert_(isinstance(name, str))
        self.name = name
        self.a = a
        self.b = b
        self.tags = frozenset(tags)

    def check(self, do):
        """
        Run the function `do` on this test case, expanding arguments
        """
        do(self.a, self.b, tags=self.tags)

    def __repr__(self):
        return f'<LinalgCase: {self.name}>'