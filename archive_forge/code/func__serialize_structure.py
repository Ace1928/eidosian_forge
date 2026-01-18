from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from .block_vector import NotFullyDefinedBlockVectorError
from .block_vector import assert_block_structure as block_vector_assert_block_structure
import numpy as np
import operator
@staticmethod
def _serialize_structure(block_vector):
    """
        Parameters
        ----------
        block_vector: BlockVector

        Returns
        -------
        list
        """
    serialized_structure = list()
    for ndx in range(block_vector.nblocks):
        blk = block_vector.get_block(ndx)
        if isinstance(blk, BlockVector):
            serialized_structure.append(-1)
            serialized_structure.append(blk.nblocks)
            serialized_structure.extend(MPIBlockVector._serialize_structure(blk))
        elif isinstance(blk, MPIBlockVector):
            raise NotImplementedError('Operation not supported for MPIBlockVectors containing other MPIBlockVectors')
        elif isinstance(blk, np.ndarray):
            serialized_structure.append(-2)
            serialized_structure.append(blk.size)
        else:
            raise NotImplementedError('Unrecognized input.')
    return serialized_structure