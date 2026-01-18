import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def _expect_int(value, msg=None):
    try:
        return int(value)
    except ValueError as e:
        if msg is None:
            msg = 'Expected an int, got %s'
        raise ValueError(msg % value) from e