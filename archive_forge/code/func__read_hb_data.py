import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def _read_hb_data(content, header):
    ptr_string = ''.join([content.read(header.pointer_nbytes_full), content.readline()])
    ptr = np.fromstring(ptr_string, dtype=int, sep=' ')
    ind_string = ''.join([content.read(header.indices_nbytes_full), content.readline()])
    ind = np.fromstring(ind_string, dtype=int, sep=' ')
    val_string = ''.join([content.read(header.values_nbytes_full), content.readline()])
    val = np.fromstring(val_string, dtype=header.values_dtype, sep=' ')
    try:
        return csc_matrix((val, ind - 1, ptr - 1), shape=(header.nrows, header.ncols))
    except ValueError as e:
        raise e