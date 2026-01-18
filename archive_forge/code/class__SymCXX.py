from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
class _SymCXX(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('symcxx').NameSpace()

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)
    DenseMatrix = _DenseMatrix