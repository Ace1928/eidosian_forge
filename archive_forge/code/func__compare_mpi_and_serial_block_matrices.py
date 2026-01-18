import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def _compare_mpi_and_serial_block_matrices(self, mpi_mat, serial_mat):
    self.assertTrue(np.allclose(mpi_mat.to_local_array(), serial_mat.toarray()))
    self.assertIsInstance(mpi_mat, MPIBlockMatrix)
    rows, columns = np.nonzero(mpi_mat.ownership_mask)
    for i, j in zip(rows, columns):
        if mpi_mat.get_block(i, j) is not None:
            self.assertTrue(np.allclose(mpi_mat.get_block(i, j).toarray(), serial_mat.get_block(i, j).toarray()))
        else:
            self.assertIsNone(serial_mat.get_block(i, j))