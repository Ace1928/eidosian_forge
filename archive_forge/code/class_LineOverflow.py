import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
class LineOverflow(Warning):
    pass