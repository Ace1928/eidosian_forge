from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
def _DenseMatrix(be, *args, **kwargs):
    if len(args) == 1:
        return be.Matrix(len(args[0]), 1, args[0], **kwargs)
    else:
        nr, nc, elems = args
        return be.Matrix(nr, nc, elems, **kwargs)