from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from .block_vector import NotFullyDefinedBlockVectorError
from .block_vector import assert_block_structure as block_vector_assert_block_structure
import numpy as np
import operator
def _reverse_binary_operation_helper(self, other, operation):
    assert_block_structure(self)
    result = self.copy_structure()
    if isinstance(other, BlockVector):
        raise RuntimeError('Operation not supported by MPIBlockVector')
    elif isinstance(other, np.ndarray):
        raise RuntimeError('Operation not supported by MPIBlockVector')
    elif np.isscalar(other):
        for i in self._owned_blocks:
            result.set_block(i, operation(other, self.get_block(i)))
        return result
    else:
        raise NotImplementedError('Operation not supported by MPIBlockVector')