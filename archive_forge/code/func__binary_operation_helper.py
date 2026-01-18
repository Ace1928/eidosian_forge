from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def _binary_operation_helper(self, other, operation):
    assert_block_structure(self)
    result = BlockMatrix(self.bshape[0], self.bshape[1])
    if isinstance(other, BlockMatrix):
        assert other.bshape == self.bshape, 'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
        assert other.shape == self.shape, 'dimensions mismatch {} != {}'.format(self.shape, other.shape)
        assert_block_structure(other)
        block_indices = np.bitwise_or(self.get_block_mask(copy=False), other.get_block_mask(copy=False))
        for i, j in zip(*np.nonzero(block_indices)):
            mat1 = self.get_block(i, j)
            mat2 = other.get_block(i, j)
            if mat1 is not None and mat2 is not None:
                result.set_block(i, j, operation(mat1, mat2))
            elif mat1 is not None:
                result.set_block(i, j, operation(mat1, 0))
            elif mat2 is not None:
                result.set_block(i, j, operation(0, mat2))
        return result
    elif isspmatrix(other):
        mat = self.copy_structure()
        mat.copyfrom(other)
        return operation(self, mat)
    elif np.isscalar(other):
        for i, j in zip(*np.nonzero(self.get_block_mask(copy=False))):
            result.set_block(i, j, operation(self.get_block(i, j), other))
        return result
    else:
        return NotImplemented