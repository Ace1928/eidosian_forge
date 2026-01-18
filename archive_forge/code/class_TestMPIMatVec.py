import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
@unittest.pytest.mark.mpi
class TestMPIMatVec(unittest.TestCase):

    @classmethod
    @unittest.skipIf(SKIPTESTS, SKIPTESTS)
    def setUpClass(cls):
        pass

    def test_get_block_vector_for_dot_product_1(self):
        rank = comm.Get_rank()
        rank_ownership = np.array([[0, 1, 2], [1, 1, 2], [0, 1, 2], [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(3, rank, sub_m.copy())
        rank_ownership = np.array([0, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v)
        res = m._get_block_vector_for_dot_product(v)
        self.assertIs(res, v)

    def test_get_block_vector_for_dot_product_2(self):
        rank = comm.Get_rank()
        rank_ownership = np.array([[1, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        elif rank == 1:
            m.set_block(0, 0, sub_m.copy())
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        rank_ownership = np.array([-1, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(0, sub_v.copy())
        if rank != 0:
            v.set_block(rank, sub_v.copy())
        res = m._get_block_vector_for_dot_product(v)
        self.assertIs(res, v)

    def test_get_block_vector_for_dot_product_3(self):
        rank = comm.Get_rank()
        rank_ownership = np.array([[1, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        elif rank == 1:
            m.set_block(0, 0, sub_m.copy())
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        rank_ownership = np.array([0, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v.copy())
        res = m._get_block_vector_for_dot_product(v)
        self.assertIsNot(res, v)
        self.assertTrue(np.array_equal(res.get_block(0), sub_v))
        if rank == 0:
            self.assertIsNone(res.get_block(1))
            self.assertIsNone(res.get_block(2))
        elif rank == 1:
            self.assertTrue(np.array_equal(res.get_block(1), sub_v))
            self.assertIsNone(res.get_block(2))
        elif rank == 2:
            self.assertTrue(np.array_equal(res.get_block(2), sub_v))
            self.assertIsNone(res.get_block(1))

    def test_get_block_vector_for_dot_product_4(self):
        rank = comm.Get_rank()
        rank_ownership = np.array([[-1, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(0, 0, sub_m.copy())
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        rank_ownership = np.array([0, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v.copy())
        res = m._get_block_vector_for_dot_product(v)
        self.assertIs(res, v)

    def test_get_block_vector_for_dot_product_5(self):
        rank = comm.Get_rank()
        rank_ownership = np.array([[1, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        elif rank == 1:
            m.set_block(0, 0, sub_m.copy())
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        v = BlockVector(3)
        sub_v = np.ones(2)
        for ndx in range(3):
            v.set_block(ndx, sub_v.copy())
        res = m._get_block_vector_for_dot_product(v)
        self.assertIs(res, v)
        v_flat = v.flatten()
        res = m._get_block_vector_for_dot_product(v_flat)
        self.assertIsInstance(res, BlockVector)
        for ndx in range(3):
            block = res.get_block(ndx)
            self.assertTrue(np.array_equal(block, sub_v))

    def test_matvec_1(self):
        rank = comm.Get_rank()
        np.random.seed(0)
        orig_m = np.zeros((8, 8))
        for ndx in range(4):
            start = ndx * 2
            stop = (ndx + 1) * 2
            orig_m[start:stop, start:stop] = np.random.uniform(-10, 10, size=(2, 2))
            orig_m[start:stop, 6:8] = np.random.uniform(-10, 10, size=(2, 2))
            orig_m[6:8, start:stop] = np.random.uniform(-10, 10, size=(2, 2))
        orig_m[6:8, 6:8] = np.random.uniform(-10, 10, size=(2, 2))
        orig_v = np.random.uniform(-10, 10, size=8)
        correct_res = coo_matrix(orig_m) * orig_v
        rank_ownership = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2], [0, 1, 2, -1]])
        m = MPIBlockMatrix(4, 4, rank_ownership, comm)
        start = rank * 2
        stop = (rank + 1) * 2
        m.set_block(rank, rank, coo_matrix(orig_m[start:stop, start:stop]))
        m.set_block(rank, 3, coo_matrix(orig_m[start:stop, 6:8]))
        m.set_block(3, rank, coo_matrix(orig_m[6:8, start:stop]))
        m.set_block(3, 3, coo_matrix(orig_m[6:8, 6:8]))
        rank_ownership = np.array([0, 1, 2, -1])
        v = MPIBlockVector(4, rank_ownership, comm)
        v.set_block(rank, orig_v[start:stop])
        v.set_block(3, orig_v[6:8])
        res: MPIBlockVector = m.dot(v)
        self.assertTrue(np.allclose(correct_res, res.make_local_copy().flatten()))
        self.assertIsInstance(res, MPIBlockVector)
        self.assertTrue(np.allclose(res.get_block(rank), correct_res[start:stop]))
        self.assertTrue(np.allclose(res.get_block(3), correct_res[6:8]))
        self.assertTrue(np.allclose(res.rank_ownership, np.array([0, 1, 2, -1])))
        self.assertFalse(res.has_none)

    def test_matvec_with_block_vector(self):
        rank = comm.Get_rank()
        np.random.seed(0)
        orig_m = np.zeros((8, 8))
        for ndx in range(4):
            start = ndx * 2
            stop = (ndx + 1) * 2
            orig_m[start:stop, start:stop] = np.random.uniform(-10, 10, size=(2, 2))
            orig_m[start:stop, 6:8] = np.random.uniform(-10, 10, size=(2, 2))
            orig_m[6:8, start:stop] = np.random.uniform(-10, 10, size=(2, 2))
        orig_m[6:8, 6:8] = np.random.uniform(-10, 10, size=(2, 2))
        orig_v = np.random.uniform(-10, 10, size=8)
        correct_res = coo_matrix(orig_m) * orig_v
        rank_ownership = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2], [0, 1, 2, -1]])
        m = MPIBlockMatrix(4, 4, rank_ownership, comm)
        start = rank * 2
        stop = (rank + 1) * 2
        m.set_block(rank, rank, coo_matrix(orig_m[start:stop, start:stop]))
        m.set_block(rank, 3, coo_matrix(orig_m[start:stop, 6:8]))
        m.set_block(3, rank, coo_matrix(orig_m[6:8, start:stop]))
        m.set_block(3, 3, coo_matrix(orig_m[6:8, 6:8]))
        v = BlockVector(4)
        for ndx in range(4):
            v.set_block(ndx, np.zeros(2))
        v.copyfrom(orig_v)
        res: MPIBlockVector = m.dot(v)
        self.assertTrue(np.allclose(correct_res, res.make_local_copy().flatten()))
        self.assertIsInstance(res, MPIBlockVector)
        self.assertTrue(np.allclose(res.get_block(rank), correct_res[start:stop]))
        self.assertTrue(np.allclose(res.get_block(3), correct_res[6:8]))
        self.assertTrue(np.allclose(res.rank_ownership, np.array([0, 1, 2, -1])))
        self.assertFalse(res.has_none)

    def test_matvect_with_empty_rows(self):
        rank = comm.Get_rank()
        rank_ownership = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2], [0, 1, 2, -1]])
        m = MPIBlockMatrix(4, 4, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(rank, 3, sub_m.copy())
        m.set_row_size(3, 2)
        rank_ownership = np.array([0, 1, 2, -1])
        v = MPIBlockVector(4, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v.copy())
        v.set_block(3, sub_v.copy())
        res = m.dot(v)
        self.assertIsInstance(res, MPIBlockVector)
        self.assertTrue(np.allclose(res.get_block(rank), sub_v * 2))
        self.assertTrue(np.allclose(res.get_block(3), np.zeros(2)))
        self.assertTrue(np.allclose(res.rank_ownership, np.array([0, 1, 2, -1])))
        self.assertFalse(res.has_none)
        rank_ownership = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2], [0, -1, -1, -1]])
        m = MPIBlockMatrix(4, 4, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(rank, 3, sub_m.copy())
        m.set_row_size(3, 2)
        res = m.dot(v)
        self.assertIsInstance(res, MPIBlockVector)
        self.assertTrue(np.allclose(res.get_block(rank), sub_v * 2))
        if rank == 0:
            self.assertTrue(np.allclose(res.get_block(3), np.zeros(2)))
        self.assertTrue(np.allclose(res.rank_ownership, np.array([0, 1, 2, 0])))
        self.assertFalse(res.has_none)
        rank_ownership = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2], [-1, -1, -1, 0]])
        m = MPIBlockMatrix(4, 4, rank_ownership, comm)
        sub_m = np.array([[1, 0], [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(rank, 3, sub_m.copy())
        m.set_row_size(3, 2)
        res = m.dot(v)
        self.assertIsInstance(res, MPIBlockVector)
        self.assertTrue(np.allclose(res.get_block(rank), sub_v * 2))
        if rank == 0:
            self.assertTrue(np.allclose(res.get_block(3), np.zeros(2)))
        self.assertTrue(np.allclose(res.rank_ownership, np.array([0, 1, 2, 0])))
        self.assertFalse(res.has_none)