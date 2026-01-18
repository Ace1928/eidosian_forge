from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from .block_vector import NotFullyDefinedBlockVectorError
from .block_vector import assert_block_structure as block_vector_assert_block_structure
import numpy as np
import operator
@staticmethod
def _create_from_serialized_structure(serialized_structure, structure_ndx, result):
    """
        Parameters
        ----------
        serialized_structure: np.ndarray
        structure_ndx: int
        result: BlockVector

        Returns
        -------
        structure_ndx: int
        """
    for ndx in range(result.nblocks):
        if serialized_structure[structure_ndx] == -1:
            structure_ndx += 1
            block = BlockVector(serialized_structure[structure_ndx])
            structure_ndx += 1
            structure_ndx = MPIBlockVector._create_from_serialized_structure(serialized_structure, structure_ndx, block)
            result.set_block(ndx, block)
        elif serialized_structure[structure_ndx] == -2:
            structure_ndx += 1
            result.set_block(ndx, np.zeros(serialized_structure[structure_ndx]))
            structure_ndx += 1
        else:
            raise ValueError('Unrecognized structure')
    return structure_ndx