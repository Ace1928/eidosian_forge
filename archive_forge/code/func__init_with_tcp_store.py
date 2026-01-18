import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _init_with_tcp_store(self, n_devices, rank, host, port):
    nccl_id = None
    if rank == 0:
        self._store.run(host, port)
        nccl_id = nccl.get_unique_id()
        shifted_nccl_id = bytes([b + 128 for b in nccl_id])
        self._store_proxy['nccl_id'] = shifted_nccl_id
        self._store_proxy.barrier()
    else:
        self._store_proxy.barrier()
        nccl_id = self._store_proxy['nccl_id']
        nccl_id = tuple([int(b) - 128 for b in nccl_id])
    self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)