import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
@classmethod
def _exchange_shape_and_sizes(cls, comm, peer, sizes_shape, method, stream):
    if comm._use_mpi:
        if method == 'send':
            sizes_shape = numpy.array(sizes_shape, dtype='q')
            comm._mpi_comm.Send(sizes_shape, dest=peer, tag=1)
            return None
        elif method == 'recv':
            sizes_shape = numpy.empty(5, dtype='q')
            comm._mpi_comm.Recv(sizes_shape, source=peer, tag=1)
            return sizes_shape
        elif method == 'bcast':
            if comm.rank == peer:
                sizes_shape = numpy.array(sizes_shape, dtype='q')
            else:
                sizes_shape = numpy.empty(5, dtype='q')
            comm._mpi_comm.Bcast(sizes_shape, root=peer)
            return sizes_shape
        elif method == 'gather':
            sizes_shape = numpy.array(sizes_shape, dtype='q')
            recv_buf = numpy.empty([comm._n_devices, 5], dtype='q')
            comm._mpi_comm.Gather(sizes_shape, recv_buf, peer)
            return recv_buf
        elif method == 'alltoall':
            sizes_shape = numpy.array(sizes_shape, dtype='q')
            recv_buf = numpy.empty([comm._n_devices, 5], dtype='q')
            comm._mpi_comm.Alltoall(sizes_shape, recv_buf)
            return recv_buf
        else:
            raise RuntimeError('Unsupported method')
    else:
        warnings.warn('Using NCCL for transferring sparse arrays metadata. This will cause device synchronization and a huge performance degradation. Please install MPI and `mpi4py` in order to avoid this issue.')
        if method == 'send':
            sizes_shape = cupy.array(sizes_shape, dtype='q')
            cls._send(comm, sizes_shape, peer, sizes_shape.dtype, 5, stream)
            return None
        elif method == 'recv':
            sizes_shape = cupy.empty(5, dtype='q')
            cls._recv(comm, sizes_shape, peer, sizes_shape.dtype, 5, stream)
            return cupy.asnumpy(sizes_shape)
        elif method == 'bcast':
            if comm.rank == peer:
                sizes_shape = cupy.array(sizes_shape, dtype='q')
            else:
                sizes_shape = cupy.empty(5, dtype='q')
            _DenseNCCLCommunicator.broadcast(comm, sizes_shape, root=peer, stream=stream)
            return cupy.asnumpy(sizes_shape)
        elif method == 'gather':
            sizes_shape = cupy.array(sizes_shape, dtype='q')
            recv_buf = cupy.empty((comm._n_devices, 5), dtype='q')
            _DenseNCCLCommunicator.gather(comm, sizes_shape, recv_buf, root=peer, stream=stream)
            return cupy.asnumpy(recv_buf)
        elif method == 'alltoall':
            sizes_shape = cupy.array(sizes_shape, dtype='q')
            recv_buf = cupy.empty((comm._n_devices, 5), dtype='q')
            _DenseNCCLCommunicator.all_to_all(comm, sizes_shape, recv_buf, stream=stream)
            return cupy.asnumpy(recv_buf)
        else:
            raise RuntimeError('Unsupported method')