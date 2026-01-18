import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
@classmethod
def _validate_field(self, field):
    if field not in self.FIELD_VALUES:
        msg = f'unknown field type {field}, must be one of {self.FIELD_VALUES}'
        raise ValueError(msg)