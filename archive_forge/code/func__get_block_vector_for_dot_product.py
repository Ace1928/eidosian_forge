from __future__ import annotations
from pyomo.common.dependencies import mpi4py
from .mpi_block_vector import MPIBlockVector
from .block_vector import BlockVector
from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
from .block_matrix import assert_block_structure as block_matrix_assert_block_structure
from .base_block import BaseBlockMatrix
import numpy as np
from scipy.sparse import coo_matrix
import operator
def _get_block_vector_for_dot_product(self, x):
    if isinstance(x, MPIBlockVector):
        '\n            Consider a non-empty block m_{i, j} from the mpi block matrix with rank owner r_m and the\n            corresponding block v_{j} from the mpi block vector with rank owner r_v. There are 4 cases:\n              1. r_m = r_v\n                 In this case, all is good.\n              2. r_v = -1\n                 In this case, all is good.\n              3. r_m = -1 and r_v = 0\n                 All is good\n              4. If none of the above cases hold, then v_{j} must be broadcast\n            '
        n_block_rows, n_block_cols = self.bshape
        blocks_needing_broadcast = np.zeros(n_block_cols, dtype=np.int64)
        x_rank_ownership = x.rank_ownership
        comm = self._mpiw
        rank = comm.Get_rank()
        if rank == 0:
            block_indices = self._owned_mask
        else:
            block_indices = self._unique_owned_mask
        block_indices = np.bitwise_and(block_indices, self.get_block_mask(copy=False))
        for i, j in zip(*np.nonzero(block_indices)):
            r_m = self._rank_owner[i, j]
            r_v = x_rank_ownership[j]
            if r_m == r_v:
                pass
            elif r_v == -1:
                pass
            elif r_m == -1 and r_v == 0:
                pass
            else:
                blocks_needing_broadcast[j] = 1
        global_blocks_needing_broadcast = np.zeros(n_block_cols, dtype=np.int64)
        comm.Allreduce(blocks_needing_broadcast, global_blocks_needing_broadcast)
        indices_needing_broadcast = np.nonzero(global_blocks_needing_broadcast)[0]
        if len(indices_needing_broadcast) == 0:
            return x
        else:
            res = BlockVector(n_block_cols)
            for ndx in np.nonzero(x.ownership_mask)[0]:
                res.set_block(ndx, x.get_block(ndx))
            for j in indices_needing_broadcast:
                j_owner = x_rank_ownership[j]
                if rank == j_owner:
                    j_size = x.get_block_size(j)
                else:
                    j_size = None
                j_size = comm.bcast(j_size, j_owner)
                if rank == j_owner:
                    data = x.get_block(j).flatten()
                else:
                    data = np.empty(j_size)
                comm.Bcast(data, j_owner)
                res.set_block(j, data)
            return res
    elif isinstance(x, BlockVector):
        return x
    elif isinstance(x, np.ndarray):
        y = BlockVector(self.bshape[1])
        for ndx, size in enumerate(self.col_block_sizes(copy=False)):
            y.set_block(ndx, np.zeros(size))
        y.copyfrom(x)
        return y
    else:
        raise NotImplementedError('Dot product is not yet supported for MPIBlockMatrix*' + str(type(x)))