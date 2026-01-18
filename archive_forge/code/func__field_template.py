import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
@staticmethod
def _field_template(field, precision):
    return {MMFile.FIELD_REAL: '%%.%ie\n' % precision, MMFile.FIELD_INTEGER: '%i\n', MMFile.FIELD_UNSIGNED: '%u\n', MMFile.FIELD_COMPLEX: '%%.%ie %%.%ie\n' % (precision, precision)}.get(field, None)