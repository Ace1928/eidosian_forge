from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
class _SymPySymEngine(_SymPy):

    def __init__(self):
        self.__sym_backend__ = __import__('sympy')
        from symengine import Lambdify
        self.Lambdify = Lambdify