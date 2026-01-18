from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings

        Returns vector of column i

        Parameters
        ----------
        i: int
            Row index

        Returns
        -------
        pyomo.contrib.pynumero.sparse BlockVector

        