import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
Test a simplex S for degeneracy (linear dependence in R^dim).

        Parameters
        ----------
        S : np.array
            Simplex with rows as vertex vectors
        proj : array, optional,
            If the projection S[1:] - S[0] is already
            computed it can be added as an optional argument.
        