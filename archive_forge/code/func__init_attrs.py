import os
import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
from scipy.sparse import coo_matrix, issparse
def _init_attrs(self, **kwargs):
    """
        Initialize each attributes with the corresponding keyword arg value
        or a default of None
        """
    attrs = self.__class__.__slots__
    public_attrs = [attr[1:] for attr in attrs]
    invalid_keys = set(kwargs.keys()) - set(public_attrs)
    if invalid_keys:
        raise ValueError('found {} invalid keyword arguments, please only\n                                use {}'.format(tuple(invalid_keys), public_attrs))
    for attr in attrs:
        setattr(self, attr, kwargs.get(attr[1:], None))