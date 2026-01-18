import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _init_with_mpi(self, n_devices, rank):
    self._mpi_comm = MPI.COMM_WORLD
    self._mpi_rank = self._mpi_comm.Get_rank()
    self._mpi_comm.Barrier()
    nccl_id = None
    if self._mpi_rank == 0:
        nccl_id = nccl.get_unique_id()
    nccl_id = self._mpi_comm.bcast(nccl_id, root=0)
    self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)